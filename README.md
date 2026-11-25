# Telecom Churn Prediction

Проект прогнозирует вероятность оттока абонентов на **30 дней**. Spark SQL очищает выгрузки соединений (CDR) и обращений в поддержку и строит признаки по истории абонента. XGBoost рассчитывает вероятность оттока, по которой абоненты ранжируются для приоритетного контакта.

**Запуск:** [Windows](#windows-powershell) · [Linux](docs/linux.md) · [macOS](docs/macos.md)

## Структура проекта

```text
├── main.py                # Команды train и score
├── data.py                # Когорта, паспорт и разметка TDG
├── spark_features.py      # Очистка и агрегация в Spark SQL
├── model.py               # Обучение XGBoost, прогноз и метрики
├── config.json            # Пути, окна и параметры модели
├── requirements.txt       # Зависимости Python
├── test_pipeline.py       # Тесты конвейера
├── Dockerfile             # Запуск в контейнере
├── architecture.md        # Схема конвейера
├── refactoring_report.md  # Отчёт по рефакторингу
└── docs/                 # Инструкции для Linux и macOS
```

`dataset/` и `artifacts/` создаются локально и исключены из Git. Для чистого клона потребуется учебный архив `tdg_generator.zip`.

## Windows (PowerShell)

### 1. Открыть проект и установить зависимости

Для локального запуска нужны **Python 3.11** и **Java 17**. Укажите `JAVA_HOME` на каталог установленной Java.

```powershell
git clone https://github.com/tgal1s/Telekom-Churn-Prediction.git
cd Telekom-Churn-Prediction
python --version
& "$env:JAVA_HOME\bin\java.exe" -version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.txt
```

### 2. Подготовить данные

Если `dataset/` уже есть, переходите к шагу 3. Для генерации положите `tdg_generator.zip` в папку загрузок.

```powershell
$project = $PWD.Path
Expand-Archive "$env:USERPROFILE\Downloads\tdg_generator.zip" "$project\.tdg-work\generator"
cd "$project\.tdg-work\generator"
& "$project\.venv\Scripts\python.exe" -m tdg --level S --seed 20260827 --out "$project\dataset"
cd "$project"
```

### 3. Запустить тесты, обучение и прогноз

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe main.py --output artifacts/reproduced train
.\.venv\Scripts\python.exe main.py --output artifacts/reproduced score --as-of 2025-12-29
```

### 4. Посмотреть результаты

```powershell
$run = (Get-Content artifacts/reproduced/latest.json -Raw | ConvertFrom-Json).run
(Get-Content "artifacts/reproduced/$run/metrics.json" -Raw | ConvertFrom-Json).test
.\.venv\Scripts\python.exe -c "import pandas as pd; print(pd.read_parquet('artifacts/reproduced/scores/score-2025-12-29.parquet').head().to_string(index=False))"
```

### Запуск в Docker

С готовым `dataset/` можно запустить конвейер из корня проекта через Docker Desktop. Python и Java входят в образ.

```powershell
docker build -t telecom-churn .
New-Item -ItemType Directory -Path artifacts -Force | Out-Null
$project = $PWD.Path
$volumes = @('-v', "${project}\dataset:/app/dataset:ro", '-v', "${project}\artifacts:/app/artifacts")
docker run --rm @volumes telecom-churn --output /app/artifacts/reproduced-docker train
docker run --rm @volumes telecom-churn --output /app/artifacts/reproduced-docker score --as-of 2025-12-29
```

## Результаты

Результаты на синтетическом наборе **TDG S**, seed `20260827`.

| Показатель | Значение |
|---|---:|
| ROC-AUC | 0,676547 |
| Average Precision | 0,161458 |
| Доля ушедших среди 10% абонентов с наибольшим риском | 14,43% |
| Доля всех случаев оттока, охваченная этой группой | 28,46% |
| Строк в тестовой выборке | 377 436 |
| Абонентов в витрине на 29 декабря 2025 года | 178 344 |

**Вывод.** В тесте среди 10% абонентов с наибольшим риском отток встречался в 2,85 раза чаще, чем в среднем по выборке. Такое ранжирование помогает выбрать, с кем связаться первым при ограниченном числе контактов.

В `artifacts/reproduced/` после обучения появятся модель, метрики, отчёт качества, статистики признаков и тестовые прогнозы. Команда `score` сохраняет итоговую витрину в `scores/score-2025-12-29.parquet`. Docker сохраняет те же типы файлов в `artifacts/reproduced-docker/`.

[Архитектура](architecture.md) · [Отчёт по рефакторингу](refactoring_report.md)
