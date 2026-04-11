#!/usr/bin/env python3
"""
Unified training script for AnyModal examples.

Usage:
    python train.py                          # uses config.yaml in current directory
    python train.py --config path/to/config.yaml
    python train.py --config config.yaml --learning_rate 1e-4 --num_epochs 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train an AnyModal multimodal model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    # Allow CLI overrides for common training params
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--projector_type", type=str, default=None,
                        help="Override projector type (linear, mlp, qformer, perceiver, c_abstractor)")
    parser.add_argument("--report_to", type=str, default=None, help="Logging: none, wandb, tensorboard")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Apply CLI overrides
    training_cfg = cfg.get("training", {})
    for key in ["learning_rate", "num_epochs", "per_device_train_batch_size", "output_dir", "fp16", "bf16", "report_to"]:
        val = getattr(args, key, None)
        if val is not None:
            training_cfg[key] = val

    if args.projector_type is not None:
        cfg.setdefault("projector", {})["type"] = args.projector_type

    # Build model
    from anymodal import (
        EncoderConfig,
        LanguageModelConfig,
        ModelConfig,
        ProjectorConfig,
        TrainingConfig,
        build_hf_trainer,
        build_model,
    )
    from anymodal.data import MultiModalDataset

    model_config = ModelConfig(
        encoder=EncoderConfig(**cfg.get("encoder", {})),
        projector=ProjectorConfig(**cfg.get("projector", {})),
        language_model=LanguageModelConfig(**cfg.get("language_model", {})),
        input_start_token=cfg.get("input_start_token", "<|input_start|>"),
        input_end_token=cfg.get("input_end_token", "<|input_end|>"),
        prompt_text=cfg.get("prompt_text", "Describe this input: "),
    )

    logger.info(f"Building model: encoder={model_config.encoder.model_name}, "
                f"projector={model_config.projector.type}, "
                f"llm={model_config.language_model.model_name}")

    processor, model = build_model(model_config)
    model.print_trainable_parameters()

    # Load datasets
    data_cfg = cfg.get("data", {})
    token = os.environ.get("HF_TOKEN")

    train_dataset = MultiModalDataset(
        dataset_name=data_cfg["dataset_name"],
        processor=processor,
        image_field=data_cfg.get("image_field", "image"),
        text_field=data_cfg.get("text_field", "text"),
        split=data_cfg.get("train_split", "train"),
        dataset_config=data_cfg.get("dataset_config"),
        subset_size=data_cfg.get("subset_size"),
        subset_ratio=data_cfg.get("subset_ratio"),
        token=token,
    )

    eval_dataset = None
    if "eval_split" in data_cfg:
        eval_dataset = MultiModalDataset(
            dataset_name=data_cfg["dataset_name"],
            processor=processor,
            image_field=data_cfg.get("image_field", "image"),
            text_field=data_cfg.get("text_field", "text"),
            split=data_cfg["eval_split"],
            dataset_config=data_cfg.get("dataset_config"),
            subset_size=data_cfg.get("eval_subset_size"),
            subset_ratio=data_cfg.get("eval_subset_ratio"),
            token=token,
        )
    elif data_cfg.get("train_val_split_ratio"):
        # Split a single dataset into train/eval
        import torch

        total = len(train_dataset)
        ratio = data_cfg["train_val_split_ratio"]
        train_size = int(total * ratio)
        eval_size = total - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(training_cfg.get("seed", 42)),
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"Eval samples: {len(eval_dataset)}")

    # Build trainer
    train_config = TrainingConfig(**training_cfg)
    trainer = build_hf_trainer(model, train_config, train_dataset, eval_dataset)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_dir = os.path.join(train_config.output_dir, "final")
    model.save_pretrained(final_dir)
    logger.info(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
