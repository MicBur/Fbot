# Verwende ein offizielles Python-Runtime als Basis-Image
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Abhängigkeitsdatei in das Arbeitsverzeichnis
COPY requirements.txt .

# Installiere die Abhängigkeiten
# --no-cache-dir, um das Image kleiner zu halten
# --default-timeout=300, um Timeout-Probleme bei langsamen Downloads zu vermeiden
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Kopiere den Rest des Anwendungscodes in das Arbeitsverzeichnis
COPY . .

# Setze Umgebungsvariablen für die Datenbankverbindung (werden von docker-compose übergeben)
ENV POSTGRES_HOST=postgres
ENV POSTGRES_DB=fbot_db
ENV POSTGRES_USER=fbot_user
ENV POSTGRES_PASSWORD=fbot_password
# ENV GEMINI_API_KEY (wird nicht direkt hier benötigt, aber ggf. für erweiterte Features)

# Definiere den Befehl, der beim Starten des Containers ausgeführt wird
CMD ["python", "main.py"]
