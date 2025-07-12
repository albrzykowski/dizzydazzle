import argparse
from unittest.mock import MagicMock

import pytest
from PIL import Image
import dizzydazzle.dizzydazzle as dd


def test_load_image_resizes(tmp_path):
    img_path = tmp_path / "large.jpg"
    Image.new("RGB", (2000, 2000)).save(img_path)
    result = dd.load_image(img_path, max_size=640)
    assert result.width <= 640
    assert result.height <= 640


def test_load_image_invalid_file(tmp_path):
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"not an image")
    with pytest.raises(Exception):
        dd.load_image(bad_path, max_size=640)


def test_edit_image_success(tmp_path, mocker):
    input_path = tmp_path / "in.jpg"
    output_path = tmp_path / "out.jpg"
    Image.new("RGB", (512, 512)).save(input_path)

    pipe = MagicMock()
    pipe.return_value.images = [Image.new("RGB", (512, 512))]
    mocker.patch("dizzydazzle.dizzydazzle.load_image", return_value=Image.new("RGB", (512, 512)))

    dd.edit_image_with_prompt(
        pipe=pipe,
        prompt="make it futuristic",
        image_path=input_path,
        output_path=output_path,
        num_inference_steps=30,
        guidance_scale=5.0,
        max_size=640
    )
    assert output_path.exists()


def test_edit_image_error_handling(tmp_path, caplog, mocker):
    input_path = tmp_path / "bad.jpg"
    output_path = tmp_path / "fail.jpg"
    Image.new("RGB", (512, 512)).save(input_path)

    mocker.patch("dizzydazzle.dizzydazzle.load_image", side_effect=RuntimeError("boom"))

    with caplog.at_level("ERROR"):
        dd.edit_image_with_prompt(
            pipe=None,
            prompt="fail",
            image_path=input_path,
            output_path=output_path,
            num_inference_steps=50,
            guidance_scale=7.5,
            max_size=640
        )

    assert "Failed to process" in caplog.text


def test_parse_args(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "prog",
        "--command", "make it shiny",
        "--input_dir", "images/in",
        "--output_dir", "images/out",
        "--num_inference_steps", "60",
        "--guidance_scale", "8.5",
        "--max_size", "1024"
    ])
    args = dd.parse_args()
    assert args.command == "make it shiny"
    assert args.num_inference_steps == 60
    assert args.guidance_scale == 8.5
    assert args.max_size == 1024


def test_process_images_input_dir_missing(tmp_path, caplog):
    args = argparse.Namespace(
        input_dir=tmp_path / "missing",
        output_dir=tmp_path / "out",
        command="test",
        num_inference_steps=50,
        guidance_scale=7.5,
        max_size=640
    )
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            dd.process_images(args)
    assert "does not exist" in caplog.text


def test_process_images_creates_output_dir(mocker, tmp_path, caplog):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"  # will be created by the function

    image_file = input_dir / "img.jpg"
    Image.new("RGB", (256, 256)).save(image_file)

    args = argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        command="style this",
        num_inference_steps=20,
        guidance_scale=6.0,
        max_size=640
    )

    # Mock pipeline
    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [Image.new("RGB", (256, 256))]
    pipe_obj = mocker.Mock()
    pipe_obj.to.return_value = mock_pipe
    pipe_obj.enable_vae_slicing.return_value = None

    mocker.patch("dizzydazzle.dizzydazzle.StableDiffusionInstructPix2PixPipeline.from_pretrained", return_value=pipe_obj)
    mocker.patch("dizzydazzle.dizzydazzle.tqdm", side_effect=lambda x, **kwargs: x)

    with caplog.at_level("WARNING"):
        dd.process_images(args)

    assert output_dir.exists()
    assert (output_dir / "img.jpg").exists()
    assert "Creating it" in caplog.text


def test_main_executes(mocker):
    mocker.patch("dizzydazzle.dizzydazzle.configure_logging")
    mocker.patch("dizzydazzle.dizzydazzle.parse_args")
    mocker.patch("dizzydazzle.dizzydazzle.process_images")

    mock_args = MagicMock()
    dd.parse_args.return_value = mock_args

    dd.main()

    dd.configure_logging.assert_called_once()
    dd.process_images.assert_called_once_with(mock_args)