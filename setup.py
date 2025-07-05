from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    # filtrujemy puste linie i komentarze
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name="dizzydazzle",
    version="0.1.0",
    author="LA",
    description="A library for batch image editing using Stable Diffusion InstructPix2Pix, capable of processing multiple images from a folder, guided by text prompts with configurable inference parameters, and runnable with or without CUDA support. ",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/albrzykowski/dizzydazzle",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
