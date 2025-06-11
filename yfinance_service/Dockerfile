FROM python:3.9-slim
WORKDIR /service
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./service /service
CMD ["python", "main.py"]