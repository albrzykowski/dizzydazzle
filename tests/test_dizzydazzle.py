import os
import types
from unittest.mock import patch, MagicMock

from PIL import Image


@patch("dizzydazzle.dizzydazzle.Image.open")
def test_load_image_creates_thumbnail(mock_open):
    mock_img = MagicMock()
    mock_img.thumbnail = MagicMock()
    mock_open.return_value.convert.return_value = mock_img

    from dizzydazzle import load_image

    result = load_image("fake_path.jpg", 100)

    mock_open.assert_called_once_with("fake_path.jpg")
    mock_open.return_value.convert.assert_called_once_with("RGB")
    mock_img.thumbnail.assert_called_once_with((100, 100), Image.Resampling.LANCZOS)
    assert result == mock_img


@patch("dizzydazzle.dizzydazzle.load_image")
def test_edit_image_with_prompt_saves_image(mock_load_image):
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    mock_load_image.return_value = mock_image

    mock_pipe = MagicMock()
    mock_pipe.return_value.images = [mock_image]

    from dizzydazzle import edit_image_with_prompt

    edit_image_with_prompt(mock_pipe, "prompt", "in.jpg", "out.jpg", 5, 7.5, 100)

    mock_load_image.assert_called_once_with("in.jpg", 100)
    mock_pipe.assert_called_once_with(prompt="prompt", image=mock_image, num_inference_steps=5, guidance_scale=7.5)
    mock_image.save.assert_called_once_with("out.jpg")


@patch("dizzydazzle.dizzydazzle.load_image")
@patch("dizzydazzle.dizzydazzle.edit_image_with_prompt")
@patch("dizzydazzle.os.listdir")
@patch("dizzydazzle.os.makedirs")
@patch("dizzydazzle.argparse.ArgumentParser.parse_args")
@patch("dizzydazzle.StableDiffusionInstructPix2PixPipeline.from_pretrained")
@patch("dizzydazzle.torch.cuda.is_available")
def test_main_calls_edit_image_with_prompt(
    mock_cuda_available,
    mock_from_pretrained,
    mock_parse_args,
    mock_makedirs,
    mock_listdir,
    mock_edit_image,
    mock_load_image
):
    mock_cuda_available.return_value = False

    mock_pipe = MagicMock()
    mock_from_pretrained.return_value.to.return_value = mock_pipe

    args = types.SimpleNamespace(
        command="test prompt",
        input_dir="input_dir",
        output_dir="output_dir",
        num_inference_steps=10,
        guidance_scale=5.0,
        max_size=128
    )
    mock_parse_args.return_value = args

    mock_listdir.return_value = ["a.png", "b.txt", "c.jpg"]

    mock_load_image.return_value = MagicMock()

    from dizzydazzle import run
    run()

    mock_makedirs.assert_called_once_with("output_dir", exist_ok=True)

    for call in mock_edit_image.call_args_list:
        args = call.args
        assert args[0] is mock_pipe
        assert args[1] == "test prompt"
        assert os.path.normpath(args[2]) in [
            os.path.normpath(os.path.join("input_dir", "a.png")),
            os.path.normpath(os.path.join("input_dir", "c.jpg"))
        ]
        assert os.path.normpath(args[3]) in [
            os.path.normpath(os.path.join("output_dir", "a.png")),
            os.path.normpath(os.path.join("output_dir", "c.jpg"))
        ]
        assert args[4] == 10
        assert args[5] == 5.0
        assert args[6] == 128

    assert mock_edit_image.call_count == 2
