# Verwende ein offizielles Python-Runtime als Basis-Image
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Abhängigkeitsdatei in das Arbeitsverzeichnis
COPY requirements.txt .

# Installiere die Abhängigkeiten
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Kopiere den Rest des Anwendungscodes in das Arbeitsverzeichnis
COPY . .

# Umgebungsvariablen werden von docker-compose übergeben
# ENV POSTGRES_HOST=postgres
# ENV POSTGRES_DB=fbot_db
# ...
# ENV ALPACA_API_KEY=...
# ENV ALPACA_SECRET_KEY=...
# ENV ALPACA_PAPER=true

# Definiere den Befehl, der beim Starten des Containers ausgeführt wird
CMD ["python", "main.py"]
