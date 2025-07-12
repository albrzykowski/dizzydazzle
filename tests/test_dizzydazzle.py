import pytest
from unittest.mock import MagicMock
import argparse

from PIL import Image
import dizzydazzle


@pytest.fixture
def test_image(tmp_path):
    image_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (1024, 1024))
    img.save(image_path)
    return image_path


def test_load_image_resizing(test_image):
    resized = dizzydazzle.load_image(test_image, max_size=640)
    assert resized.width <= 640
    assert resized.height <= 640


def test_process_images_invalid_input_dir(tmp_path, caplog):
    args = argparse.Namespace(
        input_dir="nonexistent",
        output_dir=tmp_path / "output",
        command="prompt",
        num_inference_steps=50,
        guidance_scale=7.5,
        max_size=640
    )

    with caplog.at_level("ERROR", logger="dizzydazzle.dizzydazzle"):
        with pytest.raises(SystemExit):
            dizzydazzle.process_images(args)

    assert "Input directory" in caplog.text


def test_edit_image_with_prompt_success(tmp_path, mocker):
    image_path = tmp_path / "image.jpg"
    output_path = tmp_path / "out.jpg"
    img = Image.new("RGB", (512, 512))
    img.save(image_path)

    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [img]

    mocker.patch("dizzydazzle.Image.open", return_value=img)

    dizzydazzle.edit_image_with_prompt(
        pipe=mock_pipe,
        prompt="prompt",
        image_path=image_path,
        output_path=output_path,
        num_inference_steps=50,
        guidance_scale=7.5,
        max_size=640
    )

    assert output_path.exists()


def test_output_dir_creation(mocker, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    args = mocker.Mock(
        input_dir=input_dir,
        output_dir=output_dir,
        command="prompt",
        num_inference_steps=50,
        guidance_scale=7.5,
        max_size=640
    )

    mock_pipe = mocker.patch("dizzydazzle.StableDiffusionInstructPix2PixPipeline.from_pretrained")
    mock_pipe.return_value.to.return_value = MagicMock()

    mocker.patch("dizzydazzle.tqdm", return_value=[])

    dizzydazzle.process_images(args)

    assert output_dir.exists()
