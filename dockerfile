FROM python:3.11-slim

# working directory
WORKDIR /usr/src/app

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY . .

EXPOSE 8099

# Uvicorn port 8053
CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8099"]