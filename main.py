import argparse
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch
from PIL import Image

def load_image(path: str, max_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

def edit_image_with_prompt(pipe, prompt: str, image_path: str, output_path: str, 
                           num_inference_steps: int, guidance_scale: float, max_size: int):
    image = load_image(image_path, max_size)
    edited = pipe(prompt=prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    edited.save(output_path)
    print(f"Saved edited image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch edit images with Stable Diffusion and LLM prompt.")
    parser.add_argument('--command', required=True, help='Base text prompt for LLM.')
    parser.add_argument('--input_dir', default="images/input", help='Input directory with images.')
    parser.add_argument('--output_dir', default="images/output", help='Output directory for edited images.')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps (default 50).')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale (default 7.5).')
    parser.add_argument('--max_size', type=int, default=640, help='Max image size to load (default 640).')

    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
