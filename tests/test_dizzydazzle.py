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
@patch("dizzydazzle.os.path.isdir")
def test_main_calls_edit_image_with_prompt(
    mock_isdir,
    mock_cuda_available,
    mock_from_pretrained,
    mock_parse_args,
    mock_makedirs,
    mock_listdir,
    mock_edit_image,
    mock_load_image
):
    mock_isdir.return_value = True
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


@patch("dizzydazzle.dizzydazzle.edit_image_with_prompt")
@patch("dizzydazzle.os.listdir")
@patch("dizzydazzle.torch.cuda.is_available")
@patch("dizzydazzle.StableDiffusionInstructPix2PixPipeline.from_pretrained")
@patch("dizzydazzle.dizzydazzle.logger")
@patch("dizzydazzle.argparse.ArgumentParser.parse_args")
@patch("dizzydazzle.sys.exit")
@patch("dizzydazzle.os.path.isdir")
@patch("dizzydazzle.os.makedirs")
def test_input_output_dir_checks(
    mock_makedirs,
    mock_isdir,
    mock_sys_exit,
    mock_parse_args,
    mock_logger,
    mock_from_pretrained,
    mock_cuda_available,
    mock_listdir,
    mock_edit_image,
):
    from dizzydazzle import run
    import types
    from unittest.mock import MagicMock

    args = types.SimpleNamespace(
        command="prompt",
        input_dir="input_dir",
        output_dir="output_dir",
        num_inference_steps=10,
        guidance_scale=5.0,
        max_size=128
    )
    mock_parse_args.return_value = args
    mock_cuda_available.return_value = False
    mock_from_pretrained.return_value.to.return_value = MagicMock()

    # Test: input_dir nie istnieje
    mock_isdir.side_effect = lambda path: False if "input" in path else True
    run()
    mock_logger.error.assert_called_once_with("Input directory 'input_dir' does not exist.")
    mock_sys_exit.assert_called_once_with(1)

    # Reset mocków
    mock_logger.reset_mock()
    mock_sys_exit.reset_mock()
    mock_makedirs.reset_mock()
    mock_isdir.reset_mock()
    mock_listdir.reset_mock()

    # Test: input_dir istnieje, output_dir nie istnieje
    mock_isdir.side_effect = lambda path: True if "input" in path else False
    mock_listdir.return_value = []

    run()

    mock_logger.warning.assert_called_once_with("Output directory 'output_dir' does not exist. Creating it.")

    # Sprawdź, że makedirs był wywołany dokładnie dwa razy z odpowiednimi argumentami
    assert mock_makedirs.call_count == 2
    for call_args in mock_makedirs.call_args_list:
        assert call_args == (("output_dir",), {"exist_ok": True})
