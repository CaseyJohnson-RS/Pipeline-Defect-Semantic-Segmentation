# Pipline Defect Semantic Segmentation

  

Этот репозиторий - мастерская для проведения экспериментов по семантической сегментации дефектов труб.

  

## О задаче


Необходимо перейти от задачи детекции объектов к задаче семантической сегментации. Оригинальный датасет размеченный для задачи детекции находится по [ссылке](https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset).



## Инструкция настройки проекта

  

| Tool          | Version                             |
| ------------- | ----------------------------------- |
| Python        | Python 3.10                         |
| CUDA Toolkit  | 12.8                                |
| nvcc Compiler | 12.8.61                             |
| CUDA Build    | cuda_12.8.r12.8/compiler.35404655_0 |

  
Это окружение, под которым разрабатывался проект.
Установка CUDA не является обязательным шагом, но без него всё обучение будет проходить медленно.

  

**1. Создаём виртуальное окружение**

  

```bash

py -3.10 -m venv venv

```

  

**2. Активируем окружение**

  

Для Windows

  

```cmd

venv\Scripts\activate.bat

```

  

Для Linux

  

```bash

source venv/bin/activate

```

  

**3. Обновляем pip**

  

```bash

pip install --upgrade pip

```

  

**4. Устанавливаем зависимости**

  

```bash

pip install -r requirements.txt

```

  

**5. Ставим отдельно PyTorch**

  

С ним по-сложнее, может сразу не заработать. При текущей версии CUDA пришлось переустановить PyTorch:

  

```bash

pip uninstall -y torch torchvision torchaudio

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

```

  

Проверьте версию установленной CUDA и сходите на [официальный сайт PyTorch](https://pytorch.org/get-started/locally/), чтобы скачать нужную версию библиотеки.

  

Чтобы проверить версию CUDA необходимо в командной строке выполнить:

  

```cmd

nvcc --version

```

  

**6. Скачиваем датасет**

  

[Датасет](https://www.kaggle.com/datasets/caseyjohnsonrs/pipe-defect-box-semantic-segmentation/data) - создан из боксов разметки для задачи детекции.

  

Распаковываем в директорию `datasets/` так, чтобы получилась следующая структура:

  

```bash

datasets/
|-- PipeBoxSegmentation/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |-- masks/
|   |   |-- train/
|   |   |-- val/

```

  

Названия директорий должны быть именно такими!

  

**7. Создаём переменные окружения**

  

Создаём в корне проекта файл `.env` примерно с таким наполнением:


```bash

MLFLOW_TRACKING_URI=http://111.111.111.11:1111
EXPERIMENT_NAME = "Pipeline Defects Detection"
MODELS_DIR = 'models'
DATASETS_DIR = 'datasets'
UNET_MODEL_PREFIX = "unet_bss_"
USERNAME="Casey"

```

  

**8. Проверяем, что всё работает**

Запускаем минимальный скрипт `scripts/tools/health_check.py`. Естественно, правильно ли вы настроили окружение полностью выяснится, когда перейдёте к экспериментам.

Готово!

## Как проводить эксперименты

Переходим в директорию `scripts/experiments`. В ней находятся директории под названиями экспериментов. Открываете интересующую и читаете `README.md`, который объясняет как запустить и как сконфигурировать.
