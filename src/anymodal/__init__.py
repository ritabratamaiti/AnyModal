"""
AnyModal: A flexible framework for building multimodal language models.

Quick start::

    from anymodal import ModelConfig, EncoderConfig, ProjectorConfig, build_model

    config = ModelConfig(
        encoder=EncoderConfig(model_name="google/vit-base-patch16-224"),
        projector=ProjectorConfig(type="mlp"),
        prompt_text="Describe this image: ",
    )
    processor, model = build_model(config)

    # Training
    logits, loss = model({"input": pixel_values, "text": captions})

    # Inference
    generated = model.generate(sample_input, max_new_tokens=100)
"""

from anymodal.config import (
    EncoderConfig,
    LanguageModelConfig,
    ModelConfig,
    ProjectorConfig,
    TrainingConfig,
)
from anymodal.encoders import BaseEncoder, VisionEncoder, load_vision_encoder
from anymodal.model import MultiModalModel
from anymodal.projectors import (
    BaseProjector,
    CAbstractorProjector,
    LinearProjector,
    MLPProjector,
    PerceiverProjector,
    ProjectorType,
    QFormerProjector,
    create_projector,
)
from anymodal.utils import build_hf_trainer, build_model, load_language_model

__version__ = "0.2.0"

__all__ = [
    # Model
    "MultiModalModel",
    # Config
    "ModelConfig",
    "EncoderConfig",
    "ProjectorConfig",
    "LanguageModelConfig",
    "TrainingConfig",
    # Encoders
    "BaseEncoder",
    "VisionEncoder",
    "load_vision_encoder",
    # Projectors
    "BaseProjector",
    "LinearProjector",
    "MLPProjector",
    "QFormerProjector",
    "PerceiverProjector",
    "CAbstractorProjector",
    "ProjectorType",
    "create_projector",
    # Utilities
    "build_model",
    "build_hf_trainer",
    "load_language_model",
]