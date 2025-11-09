import pytest
import torch
from unittest.mock import patch, MagicMock
from pathlib import Path

# Предположим, что функция и зависимости импортированы из модуля src.models.UNetBSS
from src.models.UNetBSS import load_unet_model

@pytest.fixture
def mock_unet():
    mock_model = MagicMock()
    mock_model.encoder.parameters.return_value = [MagicMock(requires_grad=True)]
    return mock_model

@pytest.fixture
def device():
    return torch.device("cpu")

def test_load_existing_model_success(mock_unet, device):
    mock_state_dict = {"layer": torch.tensor([1, 2, 3])}

    with (
        patch("src.models.UNetBSS.get_available_model_paths") as mock_get_paths,
        patch("src.models.UNetBSS.confirm", return_value=True),
        patch("src.models.UNetBSS.select_option", return_value="mock_model.pt"),
        patch("src.models.UNetBSS.MODELS_DIRECTORY", Path("/tmp")),
        patch("src.models.UNetBSS.Unet", return_value=mock_unet),
        patch("torch.load", return_value=mock_state_dict)
    ):
        mock_get_paths.return_value = [Path("/tmp/mock_model.pt")]

        model = load_unet_model(
            encoder_name="resnet34",
            in_channels=3,
            classes=2,
            device=device
        )

        mock_unet.load_state_dict.assert_called_once_with(mock_state_dict)
        for param in model.encoder.parameters():
            assert not param.requires_grad

def test_load_existing_model_no_selection(mock_unet, device):
    with (
        patch("src.models.UNetBSS.get_available_model_paths", return_value=[Path("/tmp/model.pt")]),
        patch("src.models.UNetBSS.confirm", return_value=True),
        patch("src.models.UNetBSS.select_option", return_value=None)
    ):
        with pytest.raises(ValueError, match="No model selected"):
            load_unet_model("resnet34", 3, 2, device)

def test_load_new_model_with_default_weights(mock_unet, device):
    with (
        patch("src.models.UNetBSS.get_available_model_paths", return_value=[]),
        patch("src.models.UNetBSS.confirm", return_value=False),
        patch("src.models.UNetBSS.Unet", return_value=mock_unet)
    ):
        model = load_unet_model(
            encoder_name="resnet34",
            in_channels=3,
            classes=2,
            device=device,
            default_encoder_weights="imagenet"
        )

        mock_unet.to.assert_called_once_with(device)
        for param in model.encoder.parameters():
            assert not param.requires_grad

def test_load_new_model_without_default_weights(mock_unet, device):
    with (
        patch("src.models.UNetBSS.get_available_model_paths", return_value=[]),
        patch("src.models.UNetBSS.confirm", return_value=False),
        patch("src.models.UNetBSS.Unet", return_value=mock_unet)
    ):
        load_unet_model(
            encoder_name="resnet34",
            in_channels=3,
            classes=2,
            device=device
        )
        mock_unet.to.assert_called_once()

def test_runtime_error_on_load_failure(device):
    with (
        patch("src.models.UNetBSS.get_available_model_paths", return_value=[Path("/tmp/model.pt")]),
        patch("src.models.UNetBSS.confirm", return_value=True),
        patch("src.models.UNetBSS.select_option", return_value="model.pt"),
        patch("src.models.UNetBSS.MODELS_DIRECTORY", Path("/tmp")),
        patch("torch.load", side_effect=RuntimeError("load failed")),
        patch("src.models.UNetBSS.Unet")
    ):
        with pytest.raises(RuntimeError, match="load failed"):
            load_unet_model("resnet34", 3, 2, device)
