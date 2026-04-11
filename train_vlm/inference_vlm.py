"""
Inference with trained VLM: Qwen2.5-1.5B + SigLIP.

Usage:
    python inference_vlm.py --image path/to/image.jpg --question "What is in this image?"
    python inference_vlm.py --image https://url/to/image.jpg --question "Describe this image"
"""

import os
import sys
import json
import argparse
from io import BytesIO

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    SiglipVisionModel,
)
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from anymodal.projectors.base import create_projector


class SigLIPEncoder(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state


def load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        import urllib.request
        data = urllib.request.urlopen(source).read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(source).convert("RGB")


@torch.inference_mode()
def generate_answer(
    image: Image.Image,
    question: str,
    vision_encoder,
    projector,
    llm,
    tokenizer,
    image_processor,
    config: dict,
    device: torch.device,
    max_new_tokens: int = 128,
) -> str:
    # Process image
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Encode + project
    vision_features = vision_encoder(pixel_values)
    projected = projector(vision_features)  # (1, num_patches, llm_dim)

    # Build prompt
    prompt = config["prompt_template"].format(question=question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_embeds = llm.get_input_embeddings()(prompt_ids)

    # Combine vision + text embeddings
    combined_embeds = torch.cat([projected, prompt_embeds], dim=1)
    attn_mask = torch.ones(1, combined_embeds.size(1), device=device)

    # Generate
    outputs = llm.generate(
        inputs_embeds=combined_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Decode (skip any prepended tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="VLM Inference")
    parser.add_argument("--model_dir", default="./vlm-qwen-siglip-vqa/final",
                        help="Path to saved model directory")
    parser.add_argument("--image", required=True, help="Path or URL to image")
    parser.add_argument("--question", required=True, help="Question about the image")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(os.path.join(args.model_dir, "vlm_config.json")) as f:
        config = json.load(f)

    print(f"Loading vision encoder: {config['vision_encoder_name']}")
    vision_encoder = SigLIPEncoder(config["vision_encoder_name"]).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(config["vision_encoder_name"])

    print(f"Loading LLM: {config['llm_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_dir, "llm_lora"), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(
        config["llm_name"], torch_dtype=torch.float16, trust_remote_code=True
    )
    llm = PeftModel.from_pretrained(
        base_llm, os.path.join(args.model_dir, "llm_lora")
    ).to(device).eval()

    print(f"Loading projector: {config['projector_type']}")
    projector = create_projector(
        projector_type=config["projector_type"],
        input_dim=config["vision_hidden_size"],
        output_dim=config["llm_hidden_size"],
        num_layers=2,
        hidden_dim=config["llm_hidden_size"],
    )
    projector.load_state_dict(
        torch.load(os.path.join(args.model_dir, "projector.pt"), map_location=device)
    )
    projector = projector.to(device).eval()

    # Load image and generate
    image = load_image(args.image)
    print(f"\nImage: {args.image} ({image.size})")
    print(f"Question: {args.question}")
    print("-" * 40)

    answer = generate_answer(
        image, args.question,
        vision_encoder, projector, llm, tokenizer, image_processor,
        config, device, args.max_tokens,
    )
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
