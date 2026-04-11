"""
Unified dataset classes for AnyModal.

Provides a configurable dataset that works with any HuggingFace image-text dataset
by specifying field names via configuration.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """Configurable dataset for image-text multimodal tasks.

    Wraps a HuggingFace dataset and maps its fields to the standard
    ``{"input": ..., "text": ...}`` format expected by ``MultiModalModel``.

    Args:
        dataset_name: HuggingFace dataset identifier.
        processor: Image processor callable (e.g., ``ViTImageProcessor``).
        image_field: Name of the image column in the dataset.
        text_field: Name of the text column. If the column contains a list,
            a random element is selected per access.
        split: Dataset split to load.
        dataset_config: Optional dataset config name (passed as ``name=`` to ``load_dataset``).
        subset_size: If set, truncate dataset to this many samples.
        subset_ratio: If set (0-1), use this fraction of the dataset.
        transform: Optional additional image transform applied after loading.
        seed: Random seed for subsetting reproducibility.
        token: HuggingFace access token for gated datasets.

    Example::

        dataset = MultiModalDataset(
            dataset_name="Mozilla/flickr30k-transformed-captions",
            processor=vit_processor,
            image_field="image",
            text_field="original_alt_text",
            split="train",
        )
        sample = dataset[0]  # {"input": {...}, "text": "a caption"}
    """

    def __init__(
        self,
        dataset_name: str,
        processor: Callable,
        image_field: str = "image",
        text_field: str = "text",
        split: str = "train",
        dataset_config: Optional[str] = None,
        subset_size: Optional[int] = None,
        subset_ratio: Optional[float] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
        token: Optional[str] = None,
    ) -> None:
        import os

        from datasets import load_dataset

        token = token or os.environ.get("HF_TOKEN")
        load_kwargs: dict[str, Any] = {"split": split}
        if dataset_config is not None:
            load_kwargs["name"] = dataset_config
        if token is not None:
            load_kwargs["token"] = token

        self.dataset = load_dataset(dataset_name, **load_kwargs)

        # Apply subsetting
        if subset_size is not None:
            self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
        elif subset_ratio is not None:
            n = int(len(self.dataset) * subset_ratio)
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.dataset)), n)
            self.dataset = self.dataset.select(indices)

        self.processor = processor
        self.image_field = image_field
        self.text_field = text_field
        self.transform = transform

        logger.info(
            f"Loaded {dataset_name} ({split}): {len(self.dataset)} samples, "
            f"image_field='{image_field}', text_field='{text_field}'"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]

        # Load and process image
        from PIL import Image

        image = item[self.image_field]
        if not isinstance(image, Image.Image):
            try:
                image = Image.open(image)
            except Exception:
                logger.warning(f"Could not open image at index {idx}, using blank image")
                image = Image.new("RGB", (224, 224))

        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        processed = self.processor(image, return_tensors="pt")
        # Remove batch dimension added by processor
        processed = {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in processed.items()
        }

        # Get text
        text = item[self.text_field]
        if isinstance(text, list):
            text = random.choice(text) if text else ""
        text = str(text)

        return {"input": processed, "text": text}


def multimodal_collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for MultiModalDataset.

    Groups individual samples into a batch format expected by MultiModalModel.

    Args:
        features: List of dicts from dataset __getitem__.

    Returns:
        Batched dict with 'input' (stacked tensors) and 'text' (list of strings).
    """
    input_keys = list(features[0]["input"].keys())
    batched_input: dict[str, Any] = {}
    for key in input_keys:
        tensors = [f["input"][key] for f in features]
        if isinstance(tensors[0], torch.Tensor):
            batched_input[key] = torch.stack(tensors)
        else:
            batched_input[key] = tensors

    texts = [f["text"] for f in features]
    return {"input": batched_input, "text": texts}
