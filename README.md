# Pipline Defect Semantic Segmentation

  

Этот репозиторий - мастерская для проведения экспериментов по семантической сегментации дефектов труб.

  

## О задаче

  

Необходимо перейти от задачи детекции объектов к задаче семантической сегментации. Оригинальный датасет размеченный для задачи детекции находится по [ссылке](https://www.kaggle.com/datasets/simplexitypipeline/pipeline-defect-dataset). Было решено соорудить монстра из двух моделей.

  

1. Модель UNet для семантической сегментации фона

2. Обученная для задачи детекции модель YOLO

  

Для обучения модели UNet были созданы два датасета:

 - [Pipline Defect Box Semantic Segmentation](https://www.kaggle.com/datasets/caseyjohnsonrs/pipe-defect-box-semantic-segmentation/data). Семантические маски этого датасета были созданы как простые заливки боксов для задачи детекции белым цветом.

 - [Pipeline Defect Semantic Segmentation](https://www.kaggle.com/datasets/caseyjohnsonrs/pipeline-defects-semantic-segmentation/data). Семантические маски размечались в ручную, но в небольшом количестве.

  
  
  

## Инструкция настройки проекта

  

| Tool          | Version                             |
| ------------- | ----------------------------------- |
| Python        | Python 3.10                         |
| CUDA Toolkit  | 12.8                                |
| nvcc Compiler | 12.8.61                             |
| CUDA Build    | cuda_12.8.r12.8/compiler.35404655_0 |

  

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

  

**6. Скачиваем два датасета**

  

[Первый датасет](https://www.kaggle.com/datasets/caseyjohnsonrs/pipe-defect-box-semantic-segmentation/data) - создан из боксов разметки для задачи детекции.

[Второй датасет](https://www.kaggle.com/datasets/caseyjohnsonrs/pipeline-defects-semantic-segmentation/data) - создан ручками.

  

Распаковываем к папку `datasets/` так, чтобы получилась следующая структура:

  

```bash

datasets/
|-- PipeBoxSegmentation/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |-- masks/
|   |   |-- train/
|   |   |-- val/
|-- PipeSegmentation/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |-- masks/
|   |   |-- train/
|   |   |-- val/

```

  

Названия директорий должны быть именно такими!

  

**7. Запускаем аугментацию**

  

Так как хранить на сервере аугментированные данные глупо, мы будем аугментировать их у себя локально. Необходимо запустить скрипт `scripts/tools/augment_segmentation_datasets.py`.

  

У вас в папке `datasets/` появится еще две директории.

  

**8. Создаём переменные окружения**

  

Создаём в корне проекта файл `.env` примерно с таким наполнением:


```bash

MLFLOW_TRACKING_URI=http://111.111.111.111:5001
EXPERIMENT_NAME = "Pipeline Defects Detection"
MODELS_DIRECTORY = 'models'
UNET_MODEL_PREFIX = "unet_bss_"

```

  

**9. Проверяем, что всё работает**

Запускаем минимальный скрипт `scripts/tools/health_check.py`. Естественно, правильно ли вы настроили окружение полностью выяснится, когда перейдёте к экспериментам.

Готово!

## Как проводить эксперименты

Переходим в директорию `scripts/experiments`. В ней находятся директории под названиями экспериментов. Открываете вас интересующую и читаем `README.md`, который объясняет как запустить и как сконфигурировать.
