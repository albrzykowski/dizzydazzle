import argparse
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image
import logging
import sys
from pathlib import Path

from tqdm import tqdm  # Install via: pip install tqdm

logger = logging.getLogger("dizzydazzle.dizzydazzle")

def load_image(path: str, max_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def run():
    parser = argparse.ArgumentParser(description="Batch edit images with Stable Diffusion and LLM prompt.")
    parser.add_argument('--command', required=True, help='Base text prompt for LLM.')
    parser.add_argument('--input_dir', default="images/input", help='Input directory with images.')
    parser.add_argument('--output_dir', default="images/output", help='Output directory for edited images.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps (default 50).')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale (default 7.5).')
    parser.add_argument('--max_size', type=int, default=640, help='Max image size to load (default 640).')

    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        logger.warning(f"Output directory '{args.output_dir}' does not exist. Creating it.")
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", safety_checker=None
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)
            edit_image_with_prompt(pipe, args.command, input_path, output_path, args.num_inference_steps, args.guidance_scale, args.max_size)

logger = logging.getLogger("dizzydazzle.dizzydazzle")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def edit_image_with_prompt(
    pipe,
    prompt: str,
    image_path: Path,
    output_path: Path,
    num_inference_steps: int,
    guidance_scale: float,
    max_size: int
):
    try:
        image = load_image(image_path, max_size)
        result = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        edited_image = result.images[0]
        edited_image.save(output_path)
        logger.info(f"Saved edited image to {output_path}")
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")


def process_images(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not output_dir.exists():
        logger.warning(f"Output directory '{output_dir}' does not exist. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        safety_checker=None
    ).to(device)

    # Performance optimization
    pipe.enable_vae_slicing()
    # pipe.enable_model_cpu_offload()  # Optional for very low VRAM systems

    supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    for image_path in tqdm(list(input_dir.iterdir()), desc="Processing images"):
        if image_path.suffix.lower() in supported_extensions:
            output_path = output_dir / image_path.name
            edit_image_with_prompt(
                pipe=pipe,
                prompt=args.command,
                image_path=image_path,
                output_path=output_path,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_size=args.max_size
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch edit images with Stable Diffusion and prompt."
    )
    parser.add_argument('--command', required=True, help='Base text prompt for editing.')
    parser.add_argument('--input_dir', default="images/input", help='Directory with input images.')
    parser.add_argument('--output_dir', default="images/output", help='Directory for edited images.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps (default 50).')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale (default 7.5).')
    parser.add_argument('--max_size', type=int, default=640, help='Max image size to load (default 640).')
    return parser.parse_args()


def main():
    configure_logging()
    args = parse_args()
    process_images(args)