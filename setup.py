from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    # filtrujemy puste linie i komentarze
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name="dizzydazzle",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A versatile image editing tool using Stable Diffusion and LLM prompts",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dizzydazzle",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
