"""Tests for the MultiModalModel core."""

import pytest
import torch
import torch.nn as nn

from anymodal.encoders.base import BaseEncoder
from anymodal.model import MultiModalModel
from anymodal.projectors import MLPProjector

# ---------------------------------------------------------------------------
# Minimal stub components for testing (no HuggingFace downloads needed)
# ---------------------------------------------------------------------------

class StubEncoder(BaseEncoder):
    """Minimal encoder that returns random features."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.dummy = nn.Linear(1, 1)  # so .parameters() is non-empty

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, inputs: dict) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        B = pixel_values.shape[0]
        return torch.randn(B, 10, self._hidden_size)


class StubConfig:
    hidden_size = 128
    eos_token_id = 2


class StubLM(nn.Module):
    """Minimal causal LM stub."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 128):
        super().__init__()
        self.config = StubConfig()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, new_size):
        old = self.embeddings
        new = nn.Embedding(new_size, old.embedding_dim)
        with torch.no_grad():
            n = min(old.num_embeddings, new_size)
            new.weight[:n] = old.weight[:n]
        self.embeddings = new
        self.head = nn.Linear(old.embedding_dim, new_size)

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        logits = self.head(inputs_embeds)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return type("Output", (), {"logits": logits, "loss": loss})()

    def generate(
        self, inputs_embeds=None, attention_mask=None,
        max_new_tokens=10, eos_token_id=None, **kwargs,
    ):
        # Return dummy token ids
        return torch.tensor([[1, 5, 3, 2]])

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    def parameters(self, recurse=True):
        yield from self.embeddings.parameters()
        yield from self.head.parameters()


class StubTokenizer:
    """Minimal tokenizer stub."""

    eos_token = "<eos>"
    pad_token = "<pad>"
    pad_token_id = 0
    _vocab = {"<eos>": 2, "<pad>": 0, "<|input_start|>": 98, "<|input_end|>": 99}
    _next_id = 100

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Simple char-level tokenization for testing
        ids = [ord(c) % 50 + 3 for c in str(text)[:20]]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def encode(self, text, return_tensors=None):
        ids = [ord(c) % 50 + 3 for c in str(text)[:20]]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "generated text"

    def add_tokens(self, tokens, special_tokens=False):
        added = 0
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = self._next_id
                self._next_id += 1
                added += 1
        return added

    def __len__(self):
        return self._next_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    encoder = StubEncoder(hidden_size=64)
    projector = MLPProjector(input_dim=64, output_dim=128)
    lm = StubLM(vocab_size=100, hidden_size=128)
    tokenizer = StubTokenizer()

    return MultiModalModel(
        encoder=encoder,
        projector=projector,
        language_model=lm,
        language_tokenizer=tokenizer,
        prompt_text="Test prompt: ",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModelInit:
    def test_creates_successfully(self, model):
        assert isinstance(model, MultiModalModel)

    def test_special_tokens_registered(self, model):
        assert model.input_start_token == "<|input_start|>"
        assert model.input_end_token == "<|input_end|>"

    def test_prompt_ids_cached(self, model):
        assert model._prompt_ids is not None
        assert model._prompt_ids.ndim == 1

    def test_print_trainable_params(self, model, capsys):
        model.print_trainable_parameters()
        captured = capsys.readouterr()
        assert "Trainable:" in captured.out


class TestForward:
    def test_returns_logits_and_loss(self, model):
        batch = {
            "input": {"pixel_values": torch.randn(2, 3, 224, 224)},
            "text": ["a cat", "a dog"],
        }
        logits, loss = model(batch)
        assert logits is not None
        assert loss is not None
        assert logits.ndim == 3  # (batch, seq, vocab)
        assert loss.ndim == 0  # scalar

    def test_does_not_mutate_input(self, model):
        texts = ["hello world", "foo bar"]
        original_texts = texts.copy()
        batch = {
            "input": {"pixel_values": torch.randn(2, 3, 224, 224)},
            "text": texts,
        }
        model(batch)
        assert texts == original_texts, "forward() must not mutate input text list"

    def test_loss_backward(self, model):
        batch = {
            "input": {"pixel_values": torch.randn(2, 3, 224, 224)},
            "text": ["test caption one", "test caption two"],
        }
        _, loss = model(batch)
        loss.backward()
        # Projector should have gradients
        for p in model.projector.parameters():
            assert p.grad is not None


class TestGenerate:
    def test_returns_string(self, model):
        sample = {"pixel_values": torch.randn(3, 224, 224)}
        result = model.generate(sample, max_new_tokens=10)
        assert isinstance(result, str)

    def test_does_not_mutate_input(self, model):
        sample = {"pixel_values": torch.randn(3, 224, 224)}
        original_shape = sample["pixel_values"].shape
        model.generate(sample, max_new_tokens=10)
        assert sample["pixel_values"].shape == original_shape, "generate() must not mutate input"


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, model, tmp_path):
        save_dir = str(tmp_path / "saved_model")

        # Save
        model.save_pretrained(save_dir)
        assert (tmp_path / "saved_model" / "projector.pt").exists()
        assert (tmp_path / "saved_model" / "config.json").exists()
        assert (tmp_path / "saved_model" / "language_model").exists()

        # Load projector weights
        state = torch.load(str(tmp_path / "saved_model" / "projector.pt"), weights_only=True)
        assert len(state) > 0

    def test_config_json_contents(self, model, tmp_path):
        import json

        save_dir = str(tmp_path / "saved_model")
        model.save_pretrained(save_dir)

        with open(tmp_path / "saved_model" / "config.json") as f:
            config = json.load(f)

        assert config["projector_class"] == "MLPProjector"
        assert config["encoder_class"] == "StubEncoder"
        assert config["prompt_text"] == "Test prompt: "
