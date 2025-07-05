
# dizzydazzle

**dizzydazzle** is a Python library for batch image editing using generative AI models guided by natural language prompts. It allows you to transform images (e.g. apply effects like fog, pixelation, color reduction) via a simple CLI.

---

## Installation

```bash
pip install dizzydazzle
```

Make sure you have Python 3.8+ and dependencies installed automatically.

---

## Usage

Run the script with command line arguments to process images in batch.

```bash
python -m dizzydazzle --command "<prompt>" [options]
```

### Default folders

- Input images: `images/input`  
- Output images: `images/output`

---

## Arguments

| Argument      | Description                                                                                  | Default             |
|---------------|----------------------------------------------------------------------------------------------|---------------------|
| `--command`   | **(required)** Text prompt describing the desired image transformation                       | —                   |
| `--input_dir` | Directory containing input images to process                                                | `images/input`       |
| `--output_dir`| Directory where edited images will be saved                                                | `images/output`      |
| `--max_size`  | Maximum dimension (width or height) to resize input images (maintains aspect ratio)         | `640`                |
| `--steps`     | Number of inference steps for the generative model (higher = better quality but slower)     | `50`                 |
| `--guidance`  | Guidance scale controlling adherence to the prompt (higher = more faithful to prompt)       | `7.5`                |

---

## Example

Apply a fog and CCTV-style effect to all images from default folder with custom size and steps:

```bash
python -m dizzydazzle --command "Add fog and CCTV effect: pixelated, limited colors, timestamp overlay." --max_size 300 --steps 30 --guidance 8
```

---

## How it works

1. Loads all images from input directory, resizing them to `max_size` max dimension.  
2. Sends each image with the prompt to a generative diffusion model pipeline (Stable Diffusion InstructPix2Pix).  
3. Saves the transformed images in the output directory with the same filenames.

---

## Notes

- Running on GPU (CUDA) is supported automatically if available.  
- CLI arguments allow easy parameter tuning without changing code.  
- Designed for batch image processing in workflows requiring automated AI-based image editing.

---

## License

MIT License