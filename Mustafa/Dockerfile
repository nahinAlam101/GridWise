FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock \
    && useradd --create-home --uid 10001 gridwise

COPY --chown=gridwise:gridwise gridwise/ ./gridwise/
COPY --chown=gridwise:gridwise main.py ./
USER gridwise
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health',timeout=2)"
CMD ["python", "main.py"]
