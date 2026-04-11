"""Tests for the data module."""

import torch

from anymodal.data import multimodal_collate_fn


class TestCollateFunction:
    def test_batches_tensors(self):
        features = [
            {"input": {"pixel_values": torch.randn(3, 224, 224)}, "text": "a cat"},
            {"input": {"pixel_values": torch.randn(3, 224, 224)}, "text": "a dog"},
        ]
        batch = multimodal_collate_fn(features)
        assert batch["input"]["pixel_values"].shape == (2, 3, 224, 224)
        assert batch["text"] == ["a cat", "a dog"]

    def test_preserves_text_order(self):
        features = [
            {"input": {"pixel_values": torch.randn(3, 32, 32)}, "text": f"text_{i}"}
            for i in range(5)
        ]
        batch = multimodal_collate_fn(features)
        assert batch["text"] == [f"text_{i}" for i in range(5)]

    def test_multiple_input_keys(self):
        features = [
            {
                "input": {
                    "pixel_values": torch.randn(3, 224, 224),
                    "attention_mask": torch.ones(197),
                },
                "text": "caption",
            },
            {
                "input": {
                    "pixel_values": torch.randn(3, 224, 224),
                    "attention_mask": torch.ones(197),
                },
                "text": "another caption",
            },
        ]
        batch = multimodal_collate_fn(features)
        assert batch["input"]["pixel_values"].shape == (2, 3, 224, 224)
        assert batch["input"]["attention_mask"].shape == (2, 197)


class TestConfigImports:
    """Test that all public API imports work."""

    def test_top_level_imports(self):
        pass

    def test_config_defaults(self):
        from anymodal import ModelConfig, TrainingConfig

        mc = ModelConfig()
        assert mc.encoder.model_name == "google/vit-base-patch16-224"
        assert mc.projector.type == "mlp"
        assert mc.language_model.model_name == "meta-llama/Llama-3.2-1B"

        tc = TrainingConfig()
        assert tc.num_epochs == 3
        assert tc.learning_rate == 3e-4
        assert tc.lr_scheduler_type == "cosine"
        assert tc.max_grad_norm == 1.0
