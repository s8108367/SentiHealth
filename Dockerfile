FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]