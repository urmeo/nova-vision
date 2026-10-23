FROM python:3.12-slim

WORKDIR /app

ENV NOVA_PUBLIC=1

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir --index-url "$TORCH_INDEX" "torch==2.14.1"

COPY requirements.txt README.md LICENSE ./
COPY novavision/*.py ./novavision/
COPY novavision/affect/*.py ./novavision/affect/
COPY novavision/data/*.py novavision/data/*.txt novavision/data/*.tsv ./novavision/data/
COPY novavision/eval/*.py ./novavision/eval/
COPY novavision/experiments/*.py ./novavision/experiments/
COPY novavision/generation/*.py ./novavision/generation/
# hadolint ignore=DL3013
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY docs/ ./docs/
COPY figma/ ./figma/

RUN useradd --create-home --uid 10001 nova
USER 10001

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + (os.getenv('PORT') or '7860') + '/', timeout=3).close()"]

CMD ["sh", "-c", "exec python -m gunicorn --workers 1 --threads 4 --timeout 300 --bind \"0.0.0.0:${PORT:-7860}\" 'novavision.web:create_app()'"]
