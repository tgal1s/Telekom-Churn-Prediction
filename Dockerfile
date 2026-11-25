FROM scratch AS wheelhouse
FROM eclipse-temurin:17-jre AS java
FROM python:3.11-slim
WORKDIR /app
COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    PYTHONUNBUFFERED=1
ARG PIP_NO_INDEX=0
COPY requirements.txt .
RUN --mount=type=bind,from=wheelhouse,target=/wheels \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --find-links=/wheels -r requirements.txt \
    && chmod +x /usr/local/lib/python3.11/site-packages/pyspark/bin/*
COPY main.py data.py model.py spark_features.py config.json test_pipeline.py ./
ENTRYPOINT ["python", "main.py"]
