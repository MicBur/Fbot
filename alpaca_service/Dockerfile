FROM python:3.9-slim
WORKDIR /service
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ./service /service
# Das CMD hängt davon ab, wie dieser Dienst ausgeführt werden soll (z.B. periodisches Skript)
CMD ["python", "main.py"]