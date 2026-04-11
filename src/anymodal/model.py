"""
Core MultiModalModel — the central class of AnyModal.

Composes an encoder, projector, and language model into a unified
multimodal model for training and inference.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from anymodal.encoders.base import BaseEncoder
from anymodal.projectors.base import BaseProjector

logger = logging.getLogger(__name__)


class MultiModalModel(nn.Module):
    """Flexible multimodal language model.

    Composes three components:
        1. **Encoder**: Extracts features from non-text input (e.g., vision).
        2. **Projector**: Maps encoder features into the LLM embedding space.
        3. **Language Model**: Pretrained causal LM that generates text.

    The input sequence fed to the LM during training looks like::

        [start_token] [projected_input_tokens] [end_token] [prompt_tokens] [text_tokens] [eos]

    Loss is computed only on ``[text_tokens]`` — all prefix tokens are masked with ``-100``.

    Args:
        encoder: Modality encoder (e.g., ``VisionEncoder``).
        projector: Projector mapping encoder output to LLM space.
        language_model: HuggingFace causal language model.
        language_tokenizer: HuggingFace tokenizer for the language model.
        input_start_token: Special token marking start of modality input.
            If None, defaults to ``"<|input_start|>"``.
        input_end_token: Special token marking end of modality input.
            If None, defaults to ``"<|input_end|>"``.
        prompt_text: Text prompt prepended to generation.
        input_processor: Optional callable for raw input preprocessing.

    Example::

        model = MultiModalModel(
            encoder=vision_encoder,
            projector=mlp_projector,
            language_model=llm,
            language_tokenizer=tokenizer,
            prompt_text="Describe this image: ",
        )

        # Training
        logits, loss = model(batch)

        # Inference
        text = model.generate(sample_input, max_new_tokens=100)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        projector: BaseProjector,
        language_model: nn.Module,
        language_tokenizer: Any,
        input_start_token: Optional[str] = None,
        input_end_token: Optional[str] = None,
        prompt_text: str = "Describe this input: ",
        input_processor: Optional[Any] = None,
    ) -> None:
        super().__init__()

        # Core components (registered as submodules for proper .to() / state_dict)
        self.encoder = encoder
        self.projector = projector
        self.language_model = language_model

        # Non-module attributes
        self.language_tokenizer = language_tokenizer
        self.input_processor = input_processor
        self.prompt_text = prompt_text

        # Special tokens
        self.input_start_token = input_start_token or "<|input_start|>"
        self.input_end_token = input_end_token or "<|input_end|>"
        self._register_special_tokens()

        # Cache prompt token ids (not embeddings — embeddings depend on device)
        prompt_encoding = self.language_tokenizer(self.prompt_text, return_tensors="pt")
        self._prompt_ids: torch.Tensor
        self.register_buffer("_prompt_ids", prompt_encoding["input_ids"].squeeze(0))

    def _register_special_tokens(self) -> None:
        """Add special tokens to tokenizer and resize LM embeddings."""
        new_tokens = [self.input_start_token, self.input_end_token]
        num_added = self.language_tokenizer.add_tokens(new_tokens, special_tokens=True)
        if num_added > 0:
            self.language_model.resize_token_embeddings(len(self.language_tokenizer))
            new_size = len(self.language_tokenizer)
            logger.info(f"Added {num_added} special tokens, resized embeddings to {new_size}")

    @property
    def device(self) -> torch.device:
        """Infer device from language model parameters."""
        return next(self.language_model.parameters()).device

    @property
    def lm_hidden_size(self) -> int:
        """Hidden size of the language model."""
        return self.language_model.config.hidden_size

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using the language model's embedding layer.

        Args:
            token_ids: Integer tensor of any shape.

        Returns:
            Embeddings tensor with last dim = lm_hidden_size.
        """
        return self.language_model.get_input_embeddings()(token_ids)

    def _get_special_embedding(self, token: str) -> torch.Tensor:
        """Get embedding for a special token string.

        Returns:
            Tensor of shape (num_token_pieces, hidden_size).
        """
        encoding = self.language_tokenizer(token, return_tensors="pt", add_special_tokens=False)
        ids = encoding["input_ids"]
        return self._embed_tokens(ids.to(self.device)).squeeze(0)

    def _get_prompt_embedding(self) -> torch.Tensor:
        """Get embedding for the prompt text.

        Returns:
            Tensor of shape (prompt_length, hidden_size).
        """
        return self._embed_tokens(self._prompt_ids.to(self.device))

    def _encode_input(self, modality_input: Any) -> torch.Tensor:
        """Run input through processor → encoder → projector.

        Args:
            modality_input: Raw or preprocessed modality input.

        Returns:
            Projected embeddings of shape (batch, seq_len, lm_hidden_size).
        """
        if self.input_processor is not None:
            modality_input = self.input_processor(modality_input)
        encoded = self.encoder(modality_input)
        projected = self.projector(encoded)
        return projected

    def _build_sequence(
        self,
        projected_input: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Assemble the full embedding sequence for one sample.

        Args:
            projected_input: (seq_len, hidden_size) — projected modality tokens.
            text_ids: Optional (text_len,) — tokenized text for training.

        Returns:
            Tuple of (embeddings, labels) where:
                - embeddings: (total_len, hidden_size)
                - labels: (total_len,) with -100 for non-text positions, or None for inference.
        """
        start_emb = self._get_special_embedding(self.input_start_token)
        end_emb = self._get_special_embedding(self.input_end_token)
        prompt_emb = self._get_prompt_embedding()

        parts = [start_emb, projected_input, end_emb, prompt_emb]
        prefix_len = sum(p.shape[0] for p in parts)

        if text_ids is not None:
            text_emb = self._embed_tokens(text_ids.to(self.device))
            parts.append(text_emb)

            labels = torch.cat([
                torch.full((prefix_len,), -100, dtype=torch.long, device=self.device),
                text_ids.to(self.device),
            ])
        else:
            labels = None

        embeddings = torch.cat(parts, dim=0)
        return embeddings, labels

    def forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            batch: Dict with keys:
                - ``"input"``: Modality input (dict of tensors for encoder).
                - ``"text"``: List of target text strings.

        Returns:
            Tuple of (logits, loss).
        """
        projected = self._encode_input(batch["input"])
        text_samples = batch["text"]
        batch_size = len(text_samples)
        eos_token = self.language_tokenizer.eos_token or ""

        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_attn_masks: list[torch.Tensor] = []
        max_len = 0

        for i in range(batch_size):
            text_with_eos = text_samples[i] + eos_token
            text_ids = self.language_tokenizer(
                text_with_eos, return_tensors="pt", add_special_tokens=False
            )["input_ids"].squeeze(0)

            emb, labels = self._build_sequence(projected[i], text_ids)
            all_embeddings.append(emb)
            all_labels.append(labels)
            all_attn_masks.append(torch.ones(emb.shape[0], device=self.device))
            max_len = max(max_len, emb.shape[0])

        # Pad to max length
        pad_token_id = (
            self.language_tokenizer.pad_token_id or self.language_model.config.eos_token_id
        )
        pad_embedding = self._embed_tokens(
            torch.tensor([pad_token_id], device=self.device)
        ).squeeze(0)  # (hidden_size,)

        for i in range(batch_size):
            pad_len = max_len - all_embeddings[i].shape[0]
            if pad_len > 0:
                pad_emb = pad_embedding.unsqueeze(0).expand(pad_len, -1)
                all_embeddings[i] = torch.cat([all_embeddings[i], pad_emb], dim=0)
                all_labels[i] = torch.cat([
                    all_labels[i],
                    torch.full((pad_len,), -100, dtype=torch.long, device=self.device),
                ])
                all_attn_masks[i] = torch.cat([
                    all_attn_masks[i],
                    torch.zeros(pad_len, device=self.device),
                ])

        input_embeds = torch.stack(all_embeddings)
        labels = torch.stack(all_labels)
        attention_mask = torch.stack(all_attn_masks)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(
        self,
        modality_input: Any,
        max_new_tokens: int = 100,
        **generate_kwargs: Any,
    ) -> str:
        """Generate text from modality input.

        Args:
            modality_input: Raw modality input (single sample, not batched).
                For vision: dict with ``pixel_values`` of shape (C, H, W).
            max_new_tokens: Maximum number of tokens to generate.
            **generate_kwargs: Additional kwargs passed to ``model.generate()``.

        Returns:
            Generated text string.
        """
        # Add batch dimension without mutating caller's data
        if isinstance(modality_input, dict):
            batched = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                       for k, v in modality_input.items()}
        else:
            if isinstance(modality_input, torch.Tensor):
                batched = modality_input.unsqueeze(0)
            else:
                batched = modality_input

        projected = self._encode_input(batched)  # (1, seq, hidden)
        emb, _ = self._build_sequence(projected[0], text_ids=None)
        input_embeds = emb.unsqueeze(0)  # (1, total_len, hidden)

        output_ids = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=torch.ones(input_embeds.shape[:2], device=self.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.language_model.config.eos_token_id,
            **generate_kwargs,
        )
        return self.language_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: str) -> None:
        """Save the model components to a directory.

        Saves:
            - Projector state dict
            - Encoder state dict (if any params are trainable)
            - Language model (via save_pretrained, including PEFT adapters)
            - Model config metadata as JSON

        Args:
            output_dir: Directory path to save to.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save projector
        torch.save(self.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        logger.info(f"Saved projector to {output_dir}/projector.pt")

        # Save encoder if it has trainable params
        trainable_encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        if trainable_encoder_params:
            torch.save(self.encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))
            logger.info(f"Saved encoder to {output_dir}/encoder.pt")

        # Save language model (handles PEFT adapters automatically)
        lm_dir = os.path.join(output_dir, "language_model")
        self.language_model.save_pretrained(lm_dir)
        logger.info(f"Saved language model to {lm_dir}")

        # Save metadata
        metadata = {
            "input_start_token": self.input_start_token,
            "input_end_token": self.input_end_token,
            "prompt_text": self.prompt_text,
            "projector_class": type(self.projector).__name__,
            "projector_input_dim": self.projector.input_dim,
            "projector_output_dim": self.projector.output_dim,
            "encoder_class": type(self.encoder).__name__,
            "encoder_hidden_size": self.encoder.hidden_size,
            "lm_hidden_size": self.lm_hidden_size,
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved config to {output_dir}/config.json")

    def load_pretrained(self, model_dir: str) -> None:
        """Load saved model components from a directory.

        Args:
            model_dir: Directory containing saved model files.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        # Load projector
        projector_path = os.path.join(model_dir, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector not found at {projector_path}")
        state = torch.load(projector_path, map_location="cpu", weights_only=True)
        self.projector.load_state_dict(state)
        logger.info("Loaded projector weights")

        # Load encoder if saved
        encoder_path = os.path.join(model_dir, "encoder.pt")
        if os.path.exists(encoder_path):
            state = torch.load(encoder_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state)
            logger.info("Loaded encoder weights")

        # Load language model / PEFT adapters
        lm_dir = os.path.join(model_dir, "language_model")
        if os.path.exists(lm_dir):
            adapter_config = os.path.join(lm_dir, "adapter_config.json")
            if os.path.exists(adapter_config):
                # PEFT adapter
                self.language_model.load_adapter(lm_dir)
                logger.info("Loaded PEFT adapter")
            else:
                # Full model weights
                from transformers import AutoModelForCausalLM
                self.language_model = AutoModelForCausalLM.from_pretrained(lm_dir)
                logger.info("Loaded full language model")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return list of all trainable parameters.

        Useful for creating optimizer parameter groups.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self) -> None:
        """Print summary of trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / total if total > 0 else 0
        print(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
