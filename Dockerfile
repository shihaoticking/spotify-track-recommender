FROM python:3.9-slim AS base

WORKDIR /workspace

# Install system dependencies, including build tools
# gcc, g++, and make are needed for building hnswlib
# apt-get clean and rm -rf /var/lib/apt/lists/* are used to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./app ./app

RUN pip install --no-cache-dir -r app/requirements.txt

FROM base AS test

COPY ./tests ./tests

RUN pip install --no-cache-dir pytest

ENV PYTHONPATH="${PYTHONPATH}:."

CMD ["pytest", "-v", "tests/test_app.py"]

FROM base AS prod

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
