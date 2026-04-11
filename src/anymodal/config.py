"""
Configuration dataclasses for AnyModal models and training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EncoderConfig:
    """Configuration for the modality encoder.

    Attributes:
        model_name: HuggingFace model identifier.
        hidden_layer_index: Which hidden state layer to extract.
        freeze: Whether to freeze encoder weights.
        use_peft: Whether to apply PEFT/LoRA to the encoder.
        peft_kwargs: Additional kwargs for the PEFT config.
    """

    model_name: str = "google/vit-base-patch16-224"
    hidden_layer_index: int = -1
    freeze: bool = True
    use_peft: bool = False
    peft_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectorConfig:
    """Configuration for the projector.

    Attributes:
        type: Projector architecture: 'linear', 'mlp', 'qformer', 'perceiver', 'c_abstractor'.
        kwargs: Additional keyword arguments for the projector constructor
                (e.g., num_layers, num_queries, num_latents, etc.).
    """

    type: str = "mlp"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageModelConfig:
    """Configuration for the language model.

    Attributes:
        model_name: HuggingFace model identifier.
        use_peft: Whether to apply PEFT/LoRA.
        peft_kwargs: Kwargs for LoraConfig (r, lora_alpha, target_modules, etc.).
        quantization: Quantization type: None, '4bit', '8bit'.
        quantization_kwargs: Kwargs for BitsAndBytesConfig.
        torch_dtype: Model dtype ('float32', 'float16', 'bfloat16', 'auto').
        access_token: HuggingFace access token. Prefer HF_TOKEN env var instead.
    """

    model_name: str = "meta-llama/Llama-3.2-1B"
    use_peft: bool = False
    peft_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05,
    })
    quantization: Optional[str] = None
    quantization_kwargs: dict[str, Any] = field(default_factory=dict)
    torch_dtype: str = "auto"
    access_token: Optional[str] = None


@dataclass
class ModelConfig:
    """Top-level configuration for a MultiModalModel.

    Attributes:
        encoder: Encoder configuration.
        projector: Projector configuration.
        language_model: Language model configuration.
        input_start_token: Special token for modality input start.
        input_end_token: Special token for modality input end.
        prompt_text: Text prompt prepended to generation.
    """

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    input_start_token: str = "<|input_start|>"
    input_end_token: str = "<|input_end|>"
    prompt_text: str = "Describe this input: "


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        output_dir: Directory for checkpoints and logs.
        num_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_ratio: Fraction of total steps for LR warmup.
        lr_scheduler_type: LR scheduler type (HuggingFace scheduler names).
        max_grad_norm: Maximum gradient norm for clipping.
        fp16: Use float16 mixed precision.
        bf16: Use bfloat16 mixed precision.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        eval_strategy: When to evaluate ('no', 'epoch', 'steps').
        eval_steps: Evaluate every N steps (when eval_strategy='steps').
        save_strategy: When to save ('no', 'epoch', 'steps').
        save_steps: Save every N steps (when save_strategy='steps').
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Log every N steps.
        seed: Random seed for reproducibility.
        dataloader_num_workers: Number of data loading workers.
        report_to: Logging integrations ('none', 'wandb', 'tensorboard').
    """

    output_dir: str = "./output"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    logging_steps: int = 10
    seed: int = 42
    dataloader_num_workers: int = 4
    report_to: str = "none"
