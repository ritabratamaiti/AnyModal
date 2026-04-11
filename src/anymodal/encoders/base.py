"""
Encoder base class and vision encoder implementation.

Encoders extract feature representations from raw modality inputs.
They sit between the preprocessing step and the projector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for modality encoders.

    An encoder wraps a pretrained model that extracts feature representations
    from a specific modality (vision, audio, etc.). Subclasses must implement
    ``forward()`` and ``hidden_size``.

    The encoder should NOT manage device placement — that is handled by the
    parent MultiModalModel or the training framework.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """The dimensionality of the encoder's output features."""
        ...

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode modality inputs into feature representations.

        Args:
            inputs: Dict of tensors (e.g., {"pixel_values": tensor}).

        Returns:
            Tensor of shape (batch, seq_len, hidden_size).
        """
        ...


class VisionEncoder(BaseEncoder):
    """Wraps a HuggingFace vision model as an encoder.

    Supports any vision model that returns hidden states (ViT, SigLip, CLIP, DINOv2, etc.).
    Extracts features from a specified hidden layer.

    Args:
        model: A HuggingFace vision model (e.g., ``ViTForImageClassification``,
               ``SiglipVisionModel``).
        hidden_layer_index: Which hidden state layer to extract. Default ``-1`` (last layer).
            Use ``-2`` for penultimate layer (common with SigLip).
        freeze: If True, freeze all model parameters. Default True.

    Example::

        from transformers import ViTForImageClassification
        vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        encoder = VisionEncoder(vit, hidden_layer_index=-1, freeze=True)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_layer_index: int = -1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.hidden_layer_index = hidden_layer_index

        self._hidden_size: int = model.config.hidden_size

        if freeze:
            self.freeze()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract hidden states from the vision model.

        Args:
            inputs: Dict with at least ``pixel_values`` key.
                    Shape: (batch, channels, height, width).

        Returns:
            Hidden states tensor of shape (batch, num_patches, hidden_size).
        """
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.hidden_layer_index]
        return hidden_states


def load_vision_encoder(
    model_name: str,
    hidden_layer_index: int = -1,
    freeze: bool = True,
    use_peft: bool = False,
    peft_config: Optional[Any] = None,
) -> tuple[Any, VisionEncoder]:
    """Load a vision encoder from a HuggingFace model name.

    Automatically detects the appropriate processor and model class.

    Args:
        model_name: HuggingFace model identifier (e.g., ``"google/vit-base-patch16-224"``).
        hidden_layer_index: Hidden layer to extract features from.
        freeze: Whether to freeze encoder weights.
        use_peft: Whether to apply PEFT (LoRA) to the vision model.
        peft_config: Optional PEFT config to apply. If None and use_peft is True,
                     a sensible default LoRA config is created.

    Returns:
        Tuple of (processor, encoder) where processor is the HuggingFace image processor
        and encoder is a ``VisionEncoder`` instance.

    Example::

        processor, encoder = load_vision_encoder("google/vit-base-patch16-224")
        # processor handles image preprocessing
        # encoder.hidden_size gives the feature dimension
    """
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model

            if peft_config is None:
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["query", "value"],
                    lora_dropout=0.05,
                    bias="none",
                )
            model = get_peft_model(model, peft_config)
        except ImportError:
            raise ImportError(
                "PEFT is required for use_peft=True. Install with: pip install peft"
            )

    encoder = VisionEncoder(model, hidden_layer_index=hidden_layer_index, freeze=freeze and not use_peft)
    return processor, encoder
