FROM python:3.8

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY model/ ./model/
COPY model_deploy.py .
COPY test.csv .

CMD ["python", "model_deploy.py"]