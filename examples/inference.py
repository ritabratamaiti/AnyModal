#!/usr/bin/env python3
"""
Unified inference script for AnyModal examples.

Usage:
    python inference.py --config config.yaml --model_dir ./output/image_captioning/final
    python inference.py --config config.yaml --model_dir ./output/final --image path/to/image.jpg
    python inference.py --config config.yaml --model_dir ./output/final --num_samples 5
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained AnyModal model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of dataset samples to run")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build model
    from anymodal import (
        EncoderConfig,
        LanguageModelConfig,
        ModelConfig,
        ProjectorConfig,
        build_model,
    )

    model_config = ModelConfig(
        encoder=EncoderConfig(**cfg.get("encoder", {})),
        projector=ProjectorConfig(**cfg.get("projector", {})),
        language_model=LanguageModelConfig(**cfg.get("language_model", {})),
        input_start_token=cfg.get("input_start_token", "<|input_start|>"),
        input_end_token=cfg.get("input_end_token", "<|input_end|>"),
        prompt_text=cfg.get("prompt_text", "Describe this input: "),
    )

    processor, model = build_model(model_config)

    # Load saved weights
    logger.info(f"Loading model from {args.model_dir}")
    model.load_pretrained(args.model_dir)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.num_beams > 1:
        generate_kwargs["num_beams"] = args.num_beams

    if args.image:
        # Single image inference
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        processed = processor(image, return_tensors="pt")
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        result = model.generate(processed, **generate_kwargs)
        print(f"\nGenerated: {result}")
    else:
        # Dataset inference
        from anymodal.data import MultiModalDataset

        data_cfg = cfg.get("data", {})
        split = data_cfg.get("eval_split", data_cfg.get("train_split", "test"))
        dataset = MultiModalDataset(
            dataset_name=data_cfg["dataset_name"],
            processor=processor,
            image_field=data_cfg.get("image_field", "image"),
            text_field=data_cfg.get("text_field", "text"),
            split=split,
            dataset_config=data_cfg.get("dataset_config"),
            token=os.environ.get("HF_TOKEN"),
        )

        import random
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            result = model.generate(sample["input"], **generate_kwargs)
            print(f"\n--- Sample {i + 1} (idx={idx}) ---")
            print(f"Ground truth: {sample['text']}")
            print(f"Generated:    {result}")


if __name__ == "__main__":
    main()
