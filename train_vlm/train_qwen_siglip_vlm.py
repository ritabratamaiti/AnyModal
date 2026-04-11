"""
Train a VLM: Qwen2.5-1.5B-Instruct + SigLIP vision encoder using AnyModal.

This script fine-tunes a vision-language model that can answer questions about images.
Uses LoRA for efficient training on 2× V100 GPUs.

Usage (from outside the snap, with GPU access):
    python train_qwen_siglip_vlm.py

Or with accelerate for multi-GPU:
    accelerate launch --num_processes 2 train_qwen_siglip_vlm.py
"""

import os
import sys
import logging
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    SiglipVisionModel,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add AnyModal to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anymodal.model import MultiModalModel
from anymodal.projectors.base import create_projector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────
@dataclass
class VLMConfig:
    # Models
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    vision_encoder_name: str = "google/siglip-base-patch16-384"

    # Projector
    projector_type: str = "mlp"  # mlp works well for VLMs
    vision_hidden_size: int = 768  # SigLIP base output dim
    llm_hidden_size: int = 1536  # Qwen2.5-1.5B hidden dim

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Dataset
    dataset_name: str = "merve/vqav2-small"
    dataset_split: str = "validation"  # vqav2-small only has validation split
    max_train_samples: int = 5000
    max_eval_samples: int = 500

    # Training
    output_dir: str = "./vlm-qwen-siglip-vqa"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 4*4*2GPUs = 32
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 256
    fp16: bool = True
    dataloader_num_workers: int = 4
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 250
    seed: int = 42

    # Prompt template
    prompt_template: str = "<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
    answer_template: str = "{answer}<|im_end|>"


# ─── Vision Encoder Wrapper ────────────────────────────────────────────────
class SigLIPEncoder(torch.nn.Module):
    """Wraps SigLIP to produce (batch, seq, hidden_dim) features."""

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        # Use the sequence of patch embeddings (not CLS)
        return outputs.last_hidden_state  # (B, num_patches, hidden_dim)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size


# ─── Dataset Processing ────────────────────────────────────────────────────
class VQADataset(torch.utils.data.Dataset):
    """Visual Question Answering dataset for VLM training."""

    def __init__(self, hf_dataset, image_processor, tokenizer, config: VLMConfig):
        self.data = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        question = sample["question"]
        answer = sample.get("multiple_choice_answer", sample.get("answer", ""))

        # Process image
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Build text: prompt + answer
        prompt_text = self.config.prompt_template.format(question=question)
        full_text = prompt_text + self.config.answer_template.format(answer=answer)

        # Tokenize
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, truncation=True,
            max_length=self.config.max_seq_length // 2
        ).input_ids
        full_ids = self.tokenizer(
            full_text, add_special_tokens=False, truncation=True,
            max_length=self.config.max_seq_length
        ).input_ids

        # Labels: -100 for prompt tokens, real ids for answer tokens
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        # Ensure labels length matches input_ids
        labels = labels[: len(full_ids)]

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        }


class VLMCollator:
    """Pads batches of VLM data."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, b in enumerate(batch):
            seq_len = b["input_ids"].size(0)
            input_ids[i, :seq_len] = b["input_ids"]
            labels[i, :seq_len] = b["labels"]
            attention_mask[i, :seq_len] = b["attention_mask"]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ─── Custom Trainer ─────────────────────────────────────────────────────────
class VLMTrainer(Trainer):
    """Trainer that handles multimodal forward pass."""

    def __init__(self, vision_encoder, projector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_encoder = vision_encoder
        self.projector = projector

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs.pop("pixel_values").to(model.device)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        # 1. Encode vision
        with torch.no_grad():
            vision_features = self.vision_encoder(pixel_values)
        # 2. Project to LLM space
        projected = self.projector(vision_features)

        # 3. Get text embeddings
        text_embeds = model.get_input_embeddings()(input_ids)

        # 4. Prepend vision tokens to text
        combined_embeds = torch.cat([projected, text_embeds], dim=1)

        # 5. Extend attention mask and labels for vision tokens
        batch_size = projected.size(0)
        num_vision_tokens = projected.size(1)
        vision_attn = torch.ones(
            batch_size, num_vision_tokens, device=attention_mask.device, dtype=attention_mask.dtype
        )
        combined_attn = torch.cat([vision_attn, attention_mask], dim=1)
        vision_labels = torch.full(
            (batch_size, num_vision_tokens), -100, device=labels.device, dtype=labels.dtype
        )
        combined_labels = torch.cat([vision_labels, labels], dim=1)

        # 6. Forward through LLM
        outputs = model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attn,
            labels=combined_labels,
        )

        return (outputs.loss, outputs) if return_outputs else outputs.loss


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    config = VLMConfig()

    logger.info("=" * 60)
    logger.info("Training VLM: Qwen2.5-1.5B + SigLIP")
    logger.info("=" * 60)

    # ── 1. Load models ──
    logger.info(f"Loading vision encoder: {config.vision_encoder_name}")
    vision_encoder = SigLIPEncoder(config.vision_encoder_name, freeze=True)
    image_processor = AutoImageProcessor.from_pretrained(config.vision_encoder_name)

    logger.info(f"Loading LLM: {config.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # ── 2. Apply LoRA to LLM ──
    logger.info("Applying LoRA to LLM")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # ── 3. Create projector ──
    logger.info(f"Creating {config.projector_type} projector: {config.vision_hidden_size} → {config.llm_hidden_size}")
    projector = create_projector(
        projector_type=config.projector_type,
        input_dim=config.vision_hidden_size,
        output_dim=config.llm_hidden_size,
        num_layers=2,
        hidden_dim=config.llm_hidden_size,
    )

    # ── 4. Load dataset ──
    logger.info(f"Loading dataset: {config.dataset_name}")
    raw_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    logger.info(f"Dataset size: {len(raw_dataset)}")

    # Split into train/eval
    split = raw_dataset.train_test_split(test_size=min(config.max_eval_samples, len(raw_dataset) // 5), seed=config.seed)
    train_data = split["train"]
    eval_data = split["test"]

    if config.max_train_samples and len(train_data) > config.max_train_samples:
        train_data = train_data.select(range(config.max_train_samples))

    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_dataset = VQADataset(train_data, image_processor, tokenizer, config)
    eval_dataset = VQADataset(eval_data, image_processor, tokenizer, config)

    # ── 5. Setup training ──
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=3,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        seed=config.seed,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Move encoder and projector to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_encoder = vision_encoder.to(device)
    projector = projector.to(device)

    # ── 6. Train ──
    trainer = VLMTrainer(
        vision_encoder=vision_encoder,
        projector=projector,
        model=llm,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VLMCollator(tokenizer),
    )

    logger.info("Starting training!")
    trainer.train()

    # ── 7. Save ──
    logger.info("Saving model...")
    save_dir = os.path.join(config.output_dir, "final")
    os.makedirs(save_dir, exist_ok=True)

    # Save LoRA adapter
    llm.save_pretrained(os.path.join(save_dir, "llm_lora"))
    tokenizer.save_pretrained(os.path.join(save_dir, "llm_lora"))

    # Save projector
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pt"))

    # Save config
    import json
    with open(os.path.join(save_dir, "vlm_config.json"), "w") as f:
        json.dump({
            "llm_name": config.llm_name,
            "vision_encoder_name": config.vision_encoder_name,
            "projector_type": config.projector_type,
            "vision_hidden_size": config.vision_hidden_size,
            "llm_hidden_size": config.llm_hidden_size,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "prompt_template": config.prompt_template,
            "answer_template": config.answer_template,
        }, f, indent=2)

    logger.info(f"Model saved to {save_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
