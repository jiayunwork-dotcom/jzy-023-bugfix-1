FROM python:3.12-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HMM_DB_PATH=/data/hmm_models.sqlite3 \
    PORT=8000

EXPOSE 8000

CMD ["python", "main.py"]
