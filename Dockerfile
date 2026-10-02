FROM python:3.12-slim

LABEL org.opencontainers.image.authors="晨星"
LABEL org.opencontainers.image.source="https://github.com/CJX0712/ptqforge"

WORKDIR /app

COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Smoke test: run the full demo (trains a reference MLP, quantizes, writes JSON).
RUN python ptqforge/examples/run_demo.py --seeds 1 2 3 --datasets moons --out /tmp/benchmark.json

CMD ["python", "ptqforge/examples/run_demo.py"]
