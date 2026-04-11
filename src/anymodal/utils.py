"""
Utility functions for building and training AnyModal models.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

from anymodal.config import LanguageModelConfig, ModelConfig, TrainingConfig
from anymodal.encoders import load_vision_encoder
from anymodal.model import MultiModalModel
from anymodal.projectors import create_projector

logger = logging.getLogger(__name__)


def load_language_model(
    config: LanguageModelConfig,
) -> tuple[Any, Any]:
    """Load a language model and tokenizer from config.

    Args:
        config: Language model configuration.

    Returns:
        Tuple of (tokenizer, model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = config.access_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build model kwargs
    model_kwargs: dict[str, Any] = {"token": token}

    # Dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    if config.torch_dtype in dtype_map:
        model_kwargs["torch_dtype"] = dtype_map[config.torch_dtype]

    # Quantization
    if config.quantization in ("4bit", "8bit"):
        try:
            from transformers import BitsAndBytesConfig

            quant_kwargs = {
                "load_in_4bit": config.quantization == "4bit",
                "load_in_8bit": config.quantization == "8bit",
            }
            quant_kwargs.update(config.quantization_kwargs)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for quantization. Install with: pip install bitsandbytes"
            )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Apply PEFT
    if config.use_peft:
        try:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(**config.peft_kwargs)
            model = get_peft_model(model, peft_config)
            logger.info(f"Applied LoRA with config: {config.peft_kwargs}")
        except ImportError:
            raise ImportError("peft is required for use_peft=True. Install with: pip install peft")

    return tokenizer, model


def build_model(config: ModelConfig) -> tuple[Any, MultiModalModel]:
    """Build a complete MultiModalModel from a ModelConfig.

    This is the main entry point for constructing models.

    Args:
        config: Complete model configuration.

    Returns:
        Tuple of (processor, model) where processor handles input
        preprocessing (e.g., image processing) and model is the
        MultiModalModel ready for training or inference.

    Example::

        from anymodal import ModelConfig, build_model

        config = ModelConfig(
            encoder=EncoderConfig(model_name="google/vit-base-patch16-224"),
            projector=ProjectorConfig(type="mlp"),
            language_model=LanguageModelConfig(model_name="meta-llama/Llama-3.2-1B"),
            prompt_text="Describe this image: ",
        )
        processor, model = build_model(config)
    """
    # Load encoder
    processor, encoder = load_vision_encoder(
        model_name=config.encoder.model_name,
        hidden_layer_index=config.encoder.hidden_layer_index,
        freeze=config.encoder.freeze,
        use_peft=config.encoder.use_peft,
    )

    # Load language model
    tokenizer, language_model = load_language_model(config.language_model)

    # Create projector
    projector = create_projector(
        projector_type=config.projector.type,
        input_dim=encoder.hidden_size,
        output_dim=language_model.config.hidden_size,
        **config.projector.kwargs,
    )

    # Assemble model
    model = MultiModalModel(
        encoder=encoder,
        projector=projector,
        language_model=language_model,
        language_tokenizer=tokenizer,
        input_start_token=config.input_start_token,
        input_end_token=config.input_end_token,
        prompt_text=config.prompt_text,
    )

    return processor, model


def build_hf_trainer(
    model: MultiModalModel,
    training_config: TrainingConfig,
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
) -> Any:
    """Build a HuggingFace Trainer for the MultiModalModel.

    Creates a Trainer with a custom data collator that handles
    the multimodal sequence assembly.

    Args:
        model: The MultiModalModel to train.
        training_config: Training hyperparameters.
        train_dataset: Training dataset (must yield dicts with 'input' and 'text').
        eval_dataset: Optional evaluation dataset.

    Returns:
        A configured ``transformers.Trainer`` instance.

    Example::

        trainer = build_hf_trainer(model, training_config, train_dataset, eval_dataset)
        trainer.train()
    """
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        max_grad_norm=training_config.max_grad_norm,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        logging_steps=training_config.logging_steps,
        seed=training_config.seed,
        dataloader_num_workers=training_config.dataloader_num_workers,
        report_to=training_config.report_to,
        remove_unused_columns=False,  # we handle columns ourselves
    )

    class MultiModalCollator:
        """Custom data collator that passes batches through as-is.

        The MultiModalModel.forward() handles its own sequence assembly and padding.
        This collator groups items into the batch format expected by the model.
        """

        def __call__(self, features: list[dict]) -> dict:
            # Separate input tensors and text
            input_keys = [k for k in features[0]["input"].keys()]
            batched_input = {}
            for key in input_keys:
                tensors = [f["input"][key] for f in features]
                if isinstance(tensors[0], torch.Tensor):
                    batched_input[key] = torch.stack(tensors)
                else:
                    batched_input[key] = tensors

            texts = [f["text"] for f in features]

            return {"input": batched_input, "text": texts}

    class MultiModalTrainer(Trainer):
        """Custom Trainer that uses MultiModalModel's forward for loss computation."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            _, loss = model(inputs)
            return (loss, None) if return_outputs else loss

    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiModalCollator(),
    )

    return trainer
