"""Tests for projector architectures."""

import pytest
import torch

from anymodal.projectors import (
    CAbstractorProjector,
    LinearProjector,
    MLPProjector,
    PerceiverProjector,
    QFormerProjector,
    create_projector,
)

BATCH = 2
INPUT_DIM = 768
OUTPUT_DIM = 2048
SEQ_LEN = 197  # typical ViT: 196 patches + 1 CLS


class TestLinearProjector:
    def test_output_shape(self):
        proj = LinearProjector(INPUT_DIM, OUTPUT_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)

    def test_preserves_seq_len(self):
        proj = LinearProjector(INPUT_DIM, OUTPUT_DIM)
        for seq in [1, 50, 200]:
            x = torch.randn(1, seq, INPUT_DIM)
            assert proj(x).shape[1] == seq


class TestMLPProjector:
    def test_default_shape(self):
        proj = MLPProjector(INPUT_DIM, OUTPUT_DIM)
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)

    def test_custom_layers(self):
        proj = MLPProjector(
            INPUT_DIM, OUTPUT_DIM, hidden_dim=1024, num_layers=3, use_layer_norm=True
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)

    def test_single_layer_same_as_linear(self):
        proj = MLPProjector(INPUT_DIM, OUTPUT_DIM, num_layers=1)
        x = torch.randn(1, 10, INPUT_DIM)
        out = proj(x)
        assert out.shape == (1, 10, OUTPUT_DIM)


class TestQFormerProjector:
    def test_resamples_to_num_queries(self):
        proj = QFormerProjector(
            INPUT_DIM, OUTPUT_DIM, num_queries=32, num_layers=2, num_heads=4, head_dim=32
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, 32, OUTPUT_DIM)

    def test_different_num_queries(self):
        for nq in [8, 16, 64]:
            proj = QFormerProjector(
                INPUT_DIM, OUTPUT_DIM, num_queries=nq, num_layers=1, num_heads=4, head_dim=32
            )
            x = torch.randn(1, SEQ_LEN, INPUT_DIM)
            out = proj(x)
            assert out.shape == (1, nq, OUTPUT_DIM)

    def test_variable_input_length(self):
        proj = QFormerProjector(
            INPUT_DIM, OUTPUT_DIM, num_queries=16, num_layers=1, num_heads=4, head_dim=32
        )
        for seq in [50, 100, 300]:
            x = torch.randn(1, seq, INPUT_DIM)
            out = proj(x)
            assert out.shape == (1, 16, OUTPUT_DIM)


class TestPerceiverProjector:
    def test_resamples_to_num_latents(self):
        proj = PerceiverProjector(
            INPUT_DIM, OUTPUT_DIM, num_latents=64, num_layers=2, num_heads=4, head_dim=32
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, 64, OUTPUT_DIM)

    def test_different_num_latents(self):
        for nl in [8, 32, 128]:
            proj = PerceiverProjector(
                INPUT_DIM, OUTPUT_DIM, num_latents=nl,
                num_layers=1, num_heads=4, head_dim=32,
            )
            x = torch.randn(1, SEQ_LEN, INPUT_DIM)
            out = proj(x)
            assert out.shape == (1, nl, OUTPUT_DIM)


class TestCAbstractorProjector:
    def test_preserves_tokens_by_default(self):
        # 196 patches = 14x14 grid (perfect square)
        proj = CAbstractorProjector(INPUT_DIM, OUTPUT_DIM, num_conv_blocks=2)
        x = torch.randn(BATCH, 196, INPUT_DIM)
        out = proj(x)
        assert out.shape == (BATCH, 196, OUTPUT_DIM)

    def test_downsamples_with_output_tokens(self):
        proj = CAbstractorProjector(INPUT_DIM, OUTPUT_DIM, output_tokens=64)
        x = torch.randn(BATCH, 196, INPUT_DIM)
        out = proj(x)
        assert out.shape[0] == BATCH
        assert out.shape[2] == OUTPUT_DIM
        # Output tokens might be slightly more than 64 due to ceil(sqrt(64))=8, 8*8=64
        assert out.shape[1] == 64

    def test_non_square_input(self):
        # 200 tokens, not a perfect square — should pad
        proj = CAbstractorProjector(INPUT_DIM, OUTPUT_DIM)
        x = torch.randn(1, 200, INPUT_DIM)
        out = proj(x)
        assert out.shape[0] == 1
        assert out.shape[2] == OUTPUT_DIM
        # Padded to 15x15 = 225
        assert out.shape[1] == 225


class TestCreateProjector:
    def test_all_types(self):
        for ptype in ["linear", "mlp", "qformer", "perceiver", "c_abstractor"]:
            kwargs = {}
            if ptype == "qformer":
                kwargs = {"num_queries": 8, "num_layers": 1, "num_heads": 4, "head_dim": 32}
            elif ptype == "perceiver":
                kwargs = {"num_latents": 8, "num_layers": 1, "num_heads": 4, "head_dim": 32}
            proj = create_projector(ptype, INPUT_DIM, OUTPUT_DIM, **kwargs)
            x = torch.randn(1, 16, INPUT_DIM)
            out = proj(x)
            assert out.shape[0] == 1
            assert out.shape[2] == OUTPUT_DIM

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            create_projector("nonexistent", INPUT_DIM, OUTPUT_DIM)


class TestGradientFlow:
    """Verify gradients flow through all projectors."""

    @pytest.mark.parametrize("projector_cls,kwargs", [
        (LinearProjector, {}),
        (MLPProjector, {}),
        (QFormerProjector, {"num_queries": 8, "num_layers": 1, "num_heads": 4, "head_dim": 32}),
        (PerceiverProjector, {"num_latents": 8, "num_layers": 1, "num_heads": 4, "head_dim": 32}),
        (CAbstractorProjector, {}),
    ])
    def test_gradients_flow(self, projector_cls, kwargs):
        proj = projector_cls(INPUT_DIM, OUTPUT_DIM, **kwargs)
        x = torch.randn(1, 16, INPUT_DIM, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert all(p.grad is not None for p in proj.parameters())
