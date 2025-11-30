## Binary Semantic Segmentation (BSS)


В этом эксперименте обучается модель UNetAttention для задачи семантической сегментации фона в канализационных трубах.

### Описание датасета

Датасет состоит из изображений и боксов, в которых находятся объекты определённого класса. Классов всего 6:

0. Деформация
1. Препятствие
2. Разрыв
3. Отсоединение
4. Несоосность
5. Отложения

Анализ датасета показал, что в нём находится около 200 дубликатов изображений, и есть много изображений, которые могут запутать сеть
<p align="center">
  <img src="../../docs/photo_2025-11-30_13-12-15.jpg" width="35%" />
  <img src="../../docs/photo_2025-11-30_13-12-50.jpg" width="35%" />
  <br>
  <i>Странные изображения</i>
</p>

Так же было замечено, что есть очень схожие классы, например, разрыв очень похож на деформацию, а отсоидинение очень похоже на несоосность. Было решено на время выбросить самые малочисленные классы несоосность и разрыв.

### Задача

Получить наибольшие метрики IoU и Dice на данных, размеченных руками. Напомню их значение и формулы.

**IoU (или Jaccard Index)**

Метрика, показывающая, насколько сильно предсказанная область пересекается с истинной.

- Значение от **0 до 1**
- `1` — идеальное совпадение
- `0` — полное несовпадение

$$
IoU = \frac{|Prediction \cap GroundTruth|}{|Prediction \cup GroundTruth|}
$$

**Dice (или Sørensen–Dice coefficient)** 

Метрика измеряет схожесть двух множеств. Он чуть *мягче* к небольшим ошибкам, чем IoU.

- Значение от **0 до 1**
- `1` — идеальное совпадение
- `0` — полное несовпадение

$$
Dice = \frac{2 \times |Prediction \cap GroundTruth|}{|Prediction| + |GroundTruth|}
$$


### Конфигурация эксперимента

Скачиваем один из датасетов и распаковываем в `datasets/`:

1. [Deformation](https://drive.google.com/file/d/18vIr-kyKG3bxZd7sIsB6h_MmFDujMOxI/view?usp=sharing)
2. [Deposition](https://drive.google.com/file/d/10Q4bG7Rx7FYhgQ9oPccfx6L_dmVYo2zO/view?usp=sharing)
3. [Disconnect](https://drive.google.com/file/d/1etdRvihZfpaw2sphg85hg0wawm5kpoHJ/view?usp=sharing)
4. [Obstacle](https://drive.google.com/file/d/1c2vILrX9NTYukamw2As78rQTlWn04Noh/view?usp=sharing)

Получаем такую структуру

```text
datasets/
|-- <dataset_name>/
|   |-- images/
|   |-- masks/
|   |-- labels.csv
```

Запускаем скрипт `scripts/tools/preprocess_dataset.py` (указываем имя скачанного датасета). В директории `datasets/` появятся еще три датасета:
 - <dataset_name>_BASELINE - маленький датасет для обучения baseline модели
 - <dataset_name>_VAL - датасет для оценки baseline модели
 - <dataset_name>_EVAL - датасет для оценки всех моделей

Базовая конфигурация завершена.

### Ход и объяснение эксперимента

Базовое объяснение того, что происходит:

1. Были созданы 4 небольших датасета для предсказания какого-то отдельного класса
2. 200 картинок из каждого датасета были размечены и разделены на BASELINE и EVAL группу

Объяснение того, что будет происходить

1. Обучаем модель на BASELINE датасете. 20-40 эпох достаточно, необходимо получить высокие метрики на EVAL группе.
2. Улучшаем этой моделью плохие маски (создаётся новый датасет).
3. Обучаем нормальную модель уже на улучшенном датасете.

#### Что делать?

##### Меняем `config.yaml`

```yaml
seed: 42
experiment_name: "Pipeline Defects Detection"
track_experiment: Yes

# --- data & image -------------------------------------------------
image_size: [704, 512]
train_dataset: Deposition_BASELINE
validataion_dataset: Deposition_BASELINE_VAL
evaluation_dataset: Deposition_EVAL

# binary - for strictly binary masks
# soft - for masks in the form of probability matrices
validation_metrics: soft
evaluation_metrics: binary

# --- model --------------------------------------------------------
model:
  name: UNetAttention # For supported models check src\models\__init__.py
  args:
    encoder_name: resnet34
    encoder_weights: imagenet
    in_channels: 3
    classes: 1
    freeze_encoder: True

# --- training hyper-parameters ------------------------------------
epochs: 30
learning_rate: 0.0001
batch_size: 2

criterion:
  name: TverskyLoss # For available loss functions check src/losses/__init__.py
  args:
    alpha: 0.5   # FN
    beta: 0.5    # FP
    smooth: 1
    mode: binary
    from_logits: True

# --- logging / checkpoints ----------------------------------------
scoring_per_epoch: 4
save_by_metric: IoU
visualization_samples: 30
```

Обучаем baseline модель и сохраняем её. 

##### Создаём гибридный датасет

Запускаем `scripts/tools/make_hybrid_dataset.py`. У вас в директории `datasets/` появится еще один датасет `<dataset_name>_H<number>`. В первый раз указываем номер датасета 1, во второй раз 2 и т.д. Далее будет объяснено, зачем создавать несколько датасетов.

##### Обучаем новую (чистую) модель на модифицированных данных

Меняем в `config.yaml` строку:

```yaml
train_dataset: <dataset_name>_BASELINE # Замените на своё
``` 

на

```yaml
train_dataset: <dataset_name>_H<number> # Замените на своё
```

Запускаем обучение и играемся с параметрами.

##### Продолжение

Если хочется попробовать нечто необычное, то можно использовать только что обученную модель для очередного улучшения данных. Снова запускаем `scripts/tools/make_hybrid_dataset.py` и уже указываем последнюю модель (теперь она наш baseline) и новый номер датасета (иначе старый датасет перезапишется). Теперь у нас очередной улучшенный датасет, на котором снова можно обучать модель. Однако много итераций делать не советую, так как метрики могут упасть.
