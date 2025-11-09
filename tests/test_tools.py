import torch
import os
import random
import numpy as np
from unittest.mock import patch, MagicMock

# Импортируем тестируемые функции
from src.tools import (
    check_cuda_available,
    set_seed,
    save_model,
    check_dataset_dirs
)


# ---------- ТЕСТЫ ДЛЯ check_cuda_available ----------

@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.get_device_name", return_value="FakeGPU")
@patch("torch.version.cuda", "12.3")
def test_check_cuda_available_true(mock_avail, mock_count, mock_name):
    with patch("builtins.print") as mock_print:
        check_cuda_available()
        mock_print.assert_any_call("CUDA доступен:", True)
        mock_print.assert_any_call("Число GPU:", 1)
        mock_print.assert_any_call("Имя GPU:", "FakeGPU")
        mock_print.assert_any_call("Версия CUDA:", "12.3")


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.cuda.device_count", return_value=0)
def test_check_cuda_available_false(mock_avail, mock_count):
    with patch("builtins.print") as mock_print:
        check_cuda_available()
        mock_print.assert_any_call("CUDA доступен:", False)
        mock_print.assert_any_call("Число GPU:", 0)
        # При недоступной CUDA не должно быть попыток вывода имени устройства
        assert not any("Имя GPU:" in str(c) for c in mock_print.call_args_list)


# ---------- ТЕСТЫ ДЛЯ set_seed ----------

def test_set_seed_reproducibility():
    seed = 123
    set_seed(seed)
    # Проверим, что разные вызовы дают одинаковые результаты
    a = [random.random() for _ in range(3)]
    b = np.random.rand(3)
    torch_a = torch.rand(3)
    
    set_seed(seed)
    assert a == [random.random() for _ in range(3)]
    np.testing.assert_array_equal(b, np.random.rand(3))
    torch.testing.assert_close(torch_a, torch.rand(3))


# ---------- ТЕСТЫ ДЛЯ save_model ----------

def test_save_model_creates_file(tmp_path):
    model = MagicMock()
    model.state_dict.return_value = {"weights": torch.tensor([1, 2, 3])}

    save_path = save_model(model, tmp_path, model_name="test_model")

    # Проверим, что файл действительно создан
    assert os.path.exists(save_path)
    assert save_path.endswith("test_model.pth")

    # Проверим, что вызывался torch.save с правильными аргументами
    # (перехватим вызов через patch)
    with patch("torch.save") as mock_save:
        save_model(model, tmp_path, model_name="another")
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert isinstance(args[1], str)  # путь сохраняется как строка


# ---------- ТЕСТЫ ДЛЯ check_dataset_dirs ----------

def test_check_dataset_dirs_true(tmp_path):
    # Создаем структуру каталогов: images/{train,val}, masks/{train,val}
    for d in ["images/train", "images/val", "masks/train", "masks/val"]:
        os.makedirs(tmp_path / d)

    assert check_dataset_dirs(str(tmp_path)) is True


def test_check_dataset_dirs_false(tmp_path):
    # Пропустим одну папку, например masks/val
    for d in ["images/train", "images/val", "masks/train"]:
        os.makedirs(tmp_path / d)

    assert check_dataset_dirs(str(tmp_path)) is False
