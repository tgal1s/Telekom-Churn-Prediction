# Telecom Churn Prediction

**Spark** очищает выгрузки CDR и поддержки и строит признаки по истории абонентов. **XGBoost** прогнозирует вероятность оттока на **30 дней** и ранжирует абонентов по риску. Модель обучается на CPU.

## Установка и запуск

Для запуска через Python нужны **Python 3.11** и **Java 17**. Проверьте Java командой `java -version`. Укажите `JAVA_HOME` на каталог установки Java. В Docker Java уже входит в образ. [Требования PySpark](https://spark.apache.org/docs/4.0.1/api/python/getting_started/install.html)

В финальной локальной папке уже есть учебный датасет TDG S с seed `20260827`. После клонирования из GitHub подготовьте данные по разделу «Восстановление датасета из архива TDG» между установкой зависимостей и обучением. Каталоги `dataset/` и `artifacts/` создаются локально. Выберите блок для своей системы и укажите в первой команде путь к папке проекта.

### Windows (PowerShell)

```powershell
cd "C:\Users\Тимур\Telecom Churn Prediction"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe main.py --output artifacts/reproduced train
.\.venv\Scripts\python.exe main.py --output artifacts/reproduced score --as-of 2025-12-29
```

### Linux (Bash)

```bash
cd "$HOME/Telecom Churn Prediction"
python3.11 -m venv .venv
./.venv/bin/python -m pip install --timeout 120 -r requirements.txt
./.venv/bin/python -m unittest -v
./.venv/bin/python main.py --output artifacts/reproduced train
./.venv/bin/python main.py --output artifacts/reproduced score --as-of 2025-12-29
```

### macOS (Terminal, zsh)

Используйте нативный Python 3.11. Если XGBoost при импорте запрашивает `libomp`, установите библиотеку командой `brew install libomp`. [Документация XGBoost](https://xgboost.readthedocs.io/en/release_3.2.0/install.html)

```zsh
cd "$HOME/Telecom Churn Prediction"
python3.11 -m venv .venv
./.venv/bin/python -m pip install --timeout 120 -r requirements.txt
./.venv/bin/python -m unittest -v
./.venv/bin/python main.py --output artifacts/reproduced train
./.venv/bin/python main.py --output artifacts/reproduced score --as-of 2025-12-29
```

Тесты должны завершиться сообщением `Ran 6 tests … OK`. Команды запускают Spark и XGBoost и сохраняют результаты в `artifacts/reproduced`. В локальной финальной папке готовая модель и витрина лежат в `artifacts`. Чтобы сделать прогноз этой моделью, выполните команду `score` из своего блока без `--output artifacts/reproduced`.

## Проверка результатов

### Windows

```powershell
$run = (Get-Content artifacts/reproduced/latest.json -Raw | ConvertFrom-Json).run
(Get-Content "artifacts/reproduced/$run/metrics.json" -Raw | ConvertFrom-Json).test
.\.venv\Scripts\python.exe -c "import pandas as pd; scores = pd.read_parquet('artifacts/reproduced/scores/score-2025-12-29.parquet'); print(len(scores)); print(scores.head().to_string(index=False))"
```

### Linux и macOS

```bash
run=$(./.venv/bin/python -c "import json; print(json.load(open('artifacts/reproduced/latest.json'))['run'])")
cat "artifacts/reproduced/$run/metrics.json"
./.venv/bin/python -c "import pandas as pd; scores = pd.read_parquet('artifacts/reproduced/scores/score-2025-12-29.parquet'); print(len(scores)); print(scores.head().to_string(index=False))"
```

| Показатель | Ожидаемое значение |
|---|---:|
| ROC-AUC | 0,676547 |
| Average Precision | 0,161458 |
| Доля случаев оттока в верхних 10% по риску | 28,46% |
| Строки в тестовой выборке | 377 436 |
| Абоненты в витрине за 29 декабря 2025 года | 178 344 |

## Запуск Spark и XGBoost в Docker

Запустите Docker и выполните команды из корня проекта. Результаты будут в `artifacts/reproduced-docker`.

### Windows

```powershell
docker build -t telecom-churn .
New-Item -ItemType Directory -Path artifacts -Force | Out-Null
$project = $PWD.Path
$volumes = @('-v', "${project}\dataset:/app/dataset:ro", '-v', "${project}\artifacts:/app/artifacts")
docker run --rm @volumes telecom-churn --output /app/artifacts/reproduced-docker train
docker run --rm @volumes telecom-churn --output /app/artifacts/reproduced-docker score --as-of 2025-12-29
```

### Linux и macOS

```bash
docker build -t telecom-churn .
mkdir -p artifacts
docker run --rm -v "$PWD/dataset:/app/dataset:ro" -v "$PWD/artifacts:/app/artifacts" telecom-churn --output /app/artifacts/reproduced-docker train
docker run --rm -v "$PWD/dataset:/app/dataset:ro" -v "$PWD/artifacts:/app/artifacts" telecom-churn --output /app/artifacts/reproduced-docker score --as-of 2025-12-29
```

## Восстановление датасета из архива TDG

Этот раздел нужен для копии проекта без папки `dataset`. Выполняйте команды из корня проекта после установки зависимостей. Учебный архив `tdg_generator.zip` должен лежать в папке загрузок.

### Windows

```powershell
$project = $PWD.Path
Expand-Archive "$env:USERPROFILE\Downloads\tdg_generator.zip" "$project\.tdg-work\generator"
cd "$project\.tdg-work\generator"
& "$project\.venv\Scripts\python.exe" -m tdg --level S --seed 20260827 --out "$project\dataset"
cd $project
```

### Linux и macOS

```bash
project="$PWD"
./.venv/bin/python -m zipfile -e "$HOME/Downloads/tdg_generator.zip" .tdg-work/generator
cd .tdg-work/generator
"$project/.venv/bin/python" -m tdg --level S --seed 20260827 --out "$project/dataset"
cd "$project"
```

## Файлы проекта

- `main.py` запускает обучение и прогноз, `config.json` хранит признаки, окна и параметры модели.
- `spark_features.py` читает дневные партиции, проверяет входные файлы и вычисляет признаки в Spark SQL. Настройки локального Spark находятся в разделе `spark` файла `config.json`.
- `run-…/model.json` хранит обученную модель, `run-…/metrics.json` — метрики, `run-…/data_quality.json` — отчёт о качестве данных.
- `scores/score-2025-12-29.parquet` содержит вероятности оттока, ранги и горизонт прогноза.

XGBoost использует 150 деревьев и два семидневных окна признаков. Обучающие даты — 15 и 22 октября 2025 года. Тестовые даты — 25 и 30 ноября 2025 года. Метрики получены на синтетическом наборе.

[Архитектура](architecture.md) · [Отчёт по рефакторингу](refactoring_report.md)
