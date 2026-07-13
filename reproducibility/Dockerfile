FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV BATCH_SIZE=16
ENV NUM_WORKERS=8
ARG MODEL_DRIVE_URL="https://drive.google.com/uc?id=1PGhqwLlZHwfebCOoNJO-XqsjYkwD7Ip1"
ARG MODEL_CHECKPOINT_NAME=""
ARG MODEL_SHA256="bb3870c09ec7fa21255df1cff40b6cd2ff5a7290904562d8e43a284e4ba5c41a"

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY prepare_submission.py /app/prepare_submission.py
COPY download_model_from_drive.py /app/download_model_from_drive.py
COPY model/ /models/

RUN if [ ! -f /models/model.pt ]; then \
      MODEL_DRIVE_URL="${MODEL_DRIVE_URL}" \
      MODEL_CHECKPOINT_NAME="${MODEL_CHECKPOINT_NAME}" \
      MODEL_SHA256="${MODEL_SHA256}" \
      python /app/download_model_from_drive.py; \
    fi

RUN mkdir -p /submissions && chmod -R a+rX /app /models

ENTRYPOINT ["python", "/app/prepare_submission.py", "--data-dir", "/data", "--model-path", "/models/model.pt", "--output", "/submissions/submission.csv"]
