# Linux (Bash)

[Описание проекта и ожидаемые метрики](../README.md)

## 1. Открыть проект

Клонируйте репозиторий в выбранный каталог:

```bash
git clone https://github.com/tgal1s/Telekom-Churn-Prediction.git
cd Telekom-Churn-Prediction
```

Если папка проекта уже есть, откройте её в терминале и начните со следующего шага.

## 2. Создать окружение и установить зависимости

Для локального запуска установите Python 3.11 и Java 17. Укажите `JAVA_HOME` на установленную Java 17. Команда ниже определяет каталог по Java, выбранной в `PATH`. Для виртуального окружения нужен модуль `venv` для Python 3.11.

```bash
python3.11 --version
export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
"$JAVA_HOME/bin/java" -version
python3.11 -m venv .venv
./.venv/bin/python -m pip install --timeout 120 -r requirements.txt
```

## 3. Подготовить датасет

Если в корне проекта уже есть `dataset/`, переходите к шагу 4. Для новой копии положите учебный архив `tdg_generator.zip` в `~/Downloads` и выполните:

```bash
project="$PWD"
./.venv/bin/python -m zipfile -e "$HOME/Downloads/tdg_generator.zip" .tdg-work/generator
cd .tdg-work/generator
"$project/.venv/bin/python" -m tdg --level S --seed 20260827 --out "$project/dataset"
cd "$project"
```

## 4. Запустить тесты и обучить модель

```bash
./.venv/bin/python -m unittest -v
./.venv/bin/python main.py --output artifacts/reproduced train
```

Ожидаемый результат тестов — `Ran 6 tests` и `OK`.

## 5. Получить прогноз

```bash
./.venv/bin/python main.py --output artifacts/reproduced score --as-of 2025-12-29
```

## 6. Посмотреть метрики и прогнозы

```bash
run=$(./.venv/bin/python -c "import json; print(json.load(open('artifacts/reproduced/latest.json'))['run'])")
cat "artifacts/reproduced/$run/metrics.json"
./.venv/bin/python -c "import pandas as pd; scores = pd.read_parquet('artifacts/reproduced/scores/score-2025-12-29.parquet'); print(len(scores)); print(scores.head().to_string(index=False))"
```

## 7. Запустить через Docker

Это отдельный способ запуска. С готовым `dataset/` можно перейти к этому шагу сразу после открытия проекта. Java и Python для выполнения конвейера входят в образ.

```bash
docker build -t telecom-churn .
mkdir -p artifacts
docker run --rm -v "$PWD/dataset:/app/dataset:ro" -v "$PWD/artifacts:/app/artifacts" telecom-churn --output /app/artifacts/reproduced-docker train
docker run --rm -v "$PWD/dataset:/app/dataset:ro" -v "$PWD/artifacts:/app/artifacts" telecom-churn --output /app/artifacts/reproduced-docker score --as-of 2025-12-29
```

Результаты появятся в `artifacts/reproduced-docker/`. Для просмотра по шагу 6 замените в путях `reproduced` на `reproduced-docker`.
