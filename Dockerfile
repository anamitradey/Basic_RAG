# ---------- Dockerfile ----------
# 1) Base image: Red Hat UBI 9 with Python 3.11
FROM registry.access.redhat.com/ubi9/python-311:latest

# 2) Directory where the code will live
WORKDIR /app

# 3) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy the rest of the project (code, config.yaml, etc.)
COPY . .

# 5) Environment vars the app expects
ENV CONFIG_PATH=/app/config.yaml \
    VECTOR_STORE_PATH=/app/db \
    PYTHONUNBUFFERED=1

# 6) The port FastAPI will serve on
EXPOSE 8000/tcp

# 7) Start Uvicorn (no --reload in prod, 0.0.0.0 so it’s reachable)
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--timeout-keep-alive", "120"]
# --------------------------------
