# Toxicity Prediction

Предсказание токсичности химических соединений с использованием MLP и GCN моделей.

## Описание проекта

Проект направлен на классификацию химических соединений по уровню токсичности на основе молекулярных дескрипторов и SMILES-представлений.

### Задача

Классификация токсичности на 3 класса:

- **Класс 0 (High toxicity)**: LD50 < 50 mg/kg
- **Класс 1 (Moderate toxicity)**: LD50 = 50-500 mg/kg
- **Класс 2 (Low toxicity)**: LD50 > 500 mg/kg

### Данные

- Датасет содержит ~44,685 химических соединений
- 107 молекулярных дескрипторов после feature selection
- SMILES-представления для GCN модели
- Данные, использованные при обучении, доступны [по ссылке](https://drive.google.com/drive/folders/1xHTY-n4ranjK3vANFGJxG50SrhvZtajy?usp=share_link)

### Модели

1. **MLP (Multi-Layer Perceptron)** - работает с табличными дескрипторами
2. **GCN (Graph Convolutional Network)** - работает с молекулярными графами

## Setup

**Важно:** Проект требует Python 3.10-3.13 (onnxruntime не поддерживает Python 3.14).
Если у вас установлен Anaconda, используйте Python из conda:

```bash
# Проверьте версию Python
/opt/homebrew/anaconda3/bin/python --version  # должна быть 3.10-3.13
```

### 1. Установите Poetry (опционально)

```bash
pip install poetry
```

### 2. Клонируйте репозиторий и установите зависимости

**Вариант A: Используя Poetry (рекомендуется)**

```bash
git clone <repository-url>
cd toxicity-prediction
poetry install
```

**Вариант B: Используя pip**

```bash
git clone <repository-url>
cd toxicity-prediction
pip install -r requirements.txt
```

### 3. Активируйте виртуальное окружение

```bash
poetry shell
```

### 4. Установите pre-commit хуки

```bash
pre-commit install
```

### 5. Загрузите данные (DVC)

```bash
dvc pull
```

## Train

### Препроцессинг данных

Если нужно выполнить feature selection на сырых данных:

```bash
python -m toxicity_prediction.commands preprocess
```

Или с указанием файлов:

```bash
python -m toxicity_prediction.commands preprocess \
    --input_file mouse_external_descriptors_data_cleaned.pkl \
    --output_file data_feature_selected.pkl
```

### Обучение MLP модели

```bash
python -m toxicity_prediction.commands train-mlp
```

С кастомным файлом данных:

```bash
python -m toxicity_prediction.commands train-mlp --data_file data_feature_selected.pkl
```

### Обучение GCN модели

```bash
python -m toxicity_prediction.commands train-gcn
```

### Конфигурация

Гиперпараметры настраиваются через Hydra конфиги в папке `configs/`:

- `configs/config.yaml` - основной конфиг
- `configs/model/mlp.yaml` - параметры MLP
- `configs/model/gcn.yaml` - параметры GCN
- `configs/training/default.yaml` - параметры обучения
- `configs/data/default.yaml` - параметры данных

## Production Preparation

### Экспорт в ONNX

MLP модель автоматически экспортируется в ONNX после обучения.
Файл сохраняется в `outputs/mlp_toxicity.onnx`.

### Экспорт в TensorRT

Для конвертации ONNX модели в TensorRT:

```bash
./export_tensorrt.sh outputs/mlp_toxicity.onnx outputs/mlp_toxicity.trt
```

### Артефакты для деплоя

После обучения MLP модели в папке `outputs/` будут:

- `mlp_toxicity.onnx` - модель в формате ONNX
- `scaler.pkl` - обученный StandardScaler для нормализации признаков

Для GCN модели:

- `gcn_toxicity.pth` - веса модели PyTorch

## Infer

### Инференс MLP модели

```bash
python -m toxicity_prediction.commands infer \
    --model_path outputs/mlp_toxicity.onnx \
    --data_file data_feature_selected.pkl \
    --model_type mlp \
    --scaler_path outputs/scaler.pkl
```

### Инференс GCN модели

```bash
python -m toxicity_prediction.commands infer \
    --model_path outputs/gcn_toxicity.pth \
    --data_file data_feature_selected.pkl \
    --model_type gcn
```

### Формат входных данных

Для **MLP**: pickle файл с pandas DataFrame, содержащий 107 числовых признаков.

Для **GCN**: pickle файл с pandas DataFrame, содержащий колонки:

- `Canonical SMILES` - SMILES-представление молекулы
- `Toxicity Class Numeric` - целевой класс (0, 1, 2)

## Логирование

Метрики логируются в MLflow по адресу `http://127.0.0.1:8080`.

**Важно:** Если MLflow сервер не запущен, система автоматически переключится на CSV логирование.

Для запуска MLflow сервера локально (опционально):

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Без MLflow логи сохраняются в `outputs/logs/` в формате CSV.

### Логируемые метрики

- `train_loss` - loss на обучении
- `val_loss` - loss на валидации
- `val_roc_auc` - macro ROC-AUC на валидации
- `val_accuracy` - accuracy на валидации
- `val_f1` - macro F1 на валидации
- `test_accuracy`, `test_f1`, `test_roc_auc` - метрики на тесте

## Структура проекта

```
toxicity-prediction/
├── configs/                    # Hydra конфиги
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   └── training/
├── toxicity_prediction/        # Основной пакет
│   ├── commands.py             # CLI команды
│   ├── data/                   # Загрузка и препроцессинг
│   ├── models/                 # Архитектуры моделей
│   ├── training/               # Lightning модули
│   ├── inference/              # Инференс
│   └── export/                 # Экспорт моделей
├── plots/                      # Графики и логи
├── outputs/                    # Артефакты обучения
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

## Code Quality

Проект использует:

- **black** - форматирование кода
- **isort** - сортировка импортов
- **flake8** - линтинг
- **prettier** - форматирование YAML/JSON/MD

Запуск проверок:

```bash
pre-commit run -a
```

