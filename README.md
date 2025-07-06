
# DizzyDazzle

[![Build](https://github.com/albrzykowski/dizzydazzle/actions/workflows/tests.yml/badge.svg)](https://github.com/albrzykowski/dizzydazzle/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/github/license/albrzykowski/dizzydazzle)
![Last Commit](https://img.shields.io/github/last-commit/albrzykowski/dizzydazzle)
![Lint](https://img.shields.io/badge/linting-ruff-blue)

DizzyDazzle is a superlight  Python library for batch image editing using generative AI models guided by natural language prompts. It allows you to transform images (e.g. apply effects like fog, pixelation, color reduction) via a simple CLI. The library can also be used for image augmentation in data preprocessing pipelines, enabling synthetic variation with realistic edits.

---

## Installation

```bash
pip install git+https://github.com/albrzykowski/dizzydazzle.git
```

Make sure you have Python 3.10+ and dependencies installed automatically.

---

## Usage

Run the script with command line arguments to process images in batch.

```bash
python -m dizzydazzle --command "<prompt>" [options]
```

If the library runs correctly, you should see output similar to this:

```
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  4.70it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:34<00:00,  3.47s/it]
2025-07-06 16:49:46,403 [INFO] dizzydazzle.dizzydazzle: Saved edited image to images/output/img_1.jpg
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:38<00:00,  3.83s/it]
2025-07-06 16:49:57,467 [INFO] dizzydazzle.dizzydazzle: Saved edited image to images/output/img_2.jpg
```


### Default folders

- Input images: `images/input`  
- Output images: `images/output`

---

## Arguments

| Argument                | Description                                                                                  | Default                                                           |
|-------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `--command`             | **(required)** Text prompt describing the desired image transformation. Limited to 77 tokens (according to https://github.com/openai/CLIP) | —                   |
| `--input_dir`           | Directory containing input images to process                                                                                               | `images/input`      |
| `--output_dir`          | Directory where edited images will be saved                                                                                                | `images/output`     |
| `--max_size`            | Maximum dimension (width or height) to resize input images (maintains aspect ratio)                                                        | `640`               |
| `--num_inference_steps` | Number of inference steps for the generative model (higher = better quality but slower)                                                    | `50`                |
| `--guidance`            | Guidance scale controlling adherence to the prompt (higher = more faithful to prompt)                                                      | `7.5`               |

---

## Example

Apply a fog and CCTV-style effect to all images from default folder with custom size and steps:

```bash
python -m dizzydazzle --command "Add CCTV effect: pixelated, limited colors" --max_size 300 --num_inference_steps 30 --guidance 8
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