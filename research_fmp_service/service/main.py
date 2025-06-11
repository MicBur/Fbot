import os
import re
import time
import psycopg2
import psycopg2.extras
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from datetime import date, timedelta
import json # Hinzugefügt für das Parsen der Gemini-Antwort

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

def get_db_connection():
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432")
    return conn

def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS target_tickers (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(255),
            source VARCHAR(50),
            added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            last_sentiment_check TIMESTAMPTZ
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            data_date DATE NOT NULL,
            sentiment_score NUMERIC,
            sentiment_source VARCHAR(50),
            raw_data TEXT,
            fetched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticker) REFERENCES target_tickers(ticker) ON DELETE CASCADE,
            UNIQUE (ticker, data_date, sentiment_source)
        );
        """)
        conn.commit()
    logging.info("Tabellen 'target_tickers' und 'sentiment_data' sichergestellt.")

def get_top_us_tickers_from_gemini(api_key: str, count: int = 20) -> list[str]:
    logging.info(f"Versuche, Top {count} US-Aktien-Ticker von Gemini zu erhalten...")
    try:
        genai.configure(api_key=api_key)
        # Verwende ein aktuelles Modell, z.B. gemini-1.5-flash-latest für Geschwindigkeit oder gemini-1.5-pro-latest für Genauigkeit
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        prompt = (
            f"Liste die Ticker-Symbole der aktuellen Top {count} US-Aktien basierend auf Marktkapitalisierung und Liquidität auf. "
            "Gib das Ergebnis ausschließlich als eine einzelne Python-Liste von Börsenkürzeln (Ticker-Symbolen) zurück. "
            "Das Format sollte exakt so aussehen: [\"TICKER1\", \"TICKER2\", \"TICKER3\", ...]. "
            "Füge keine Erklärungen oder sonstigen Text hinzu."
        )
        response = model.generate_content(prompt)

        if response.parts:
            text_response = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            logging.info(f"Gemini Roh-Antwort: {text_response}")
            
            extracted_symbols = []
            try:
                # Versuche, die Antwort als Python-Liste zu parsen
                # Entferne mögliche Markdown-Code-Block-Formatierungen
                cleaned_text_response = text_response.strip()
                if cleaned_text_response.startswith("```python"):
                    cleaned_text_response = cleaned_text_response.replace("```python", "").replace("```", "").strip()
                elif cleaned_text_response.startswith("```"):
                    cleaned_text_response = cleaned_text_response.replace("```", "").strip()
                
                parsed_list = json.loads(cleaned_text_response) # json.loads kann Python-Listen-ähnliche Strings parsen
                if isinstance(parsed_list, list) and all(isinstance(s, str) for s in parsed_list):
                    extracted_symbols = [s.upper() for s in parsed_list]
            except (json.JSONDecodeError, TypeError):
                logging.warning("Konnte Gemini-Antwort nicht als JSON-Liste parsen. Fallback auf Regex.")
                # Fallback auf Regex, um Ticker-Symbole zu extrahieren
                extracted_symbols = re.findall(r'\b([A-Z]{1,5}(?:\.[A-Z])?)\b', text_response.upper())

            # Normalisierung und Duplikate entfernen
            normalized_symbols = [sym.replace("BRKB", "BRK-B").replace("BRKA", "BRK-A") for sym in extracted_symbols]
            unique_tickers = list(dict.fromkeys(normalized_symbols))
            logging.info(f"Von Gemini extrahierte Ticker: {unique_tickers[:count]}")
            return unique_tickers[:count]
        else:
            logging.warning("Keine Teile in der Gemini-Antwort gefunden.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            return []
    except Exception as e:
        logging.error(f"Fehler bei der Gemini-Anfrage: {e}")
        return []

def store_tickers(conn, tickers: list[str], source: str = "Gemini"):
    if not tickers:
        logging.info("Keine Ticker zum Speichern vorhanden.")
        return

    with conn.cursor() as cur:
        for ticker in tickers:
            try:
                cur.execute("""
                    INSERT INTO target_tickers (ticker, source)
                    VALUES (%s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        source = EXCLUDED.source, 
                        added_at = CASE 
                                    WHEN target_tickers.source IS DISTINCT FROM EXCLUDED.source THEN CURRENT_TIMESTAMP 
                                    ELSE target_tickers.added_at 
                                END;
                """, (ticker, source))
                logging.info(f"Ticker {ticker} gespeichert/aktualisiert.")
            except Exception as e:
                logging.error(f"Fehler beim Speichern von Ticker {ticker}: {e}")
                conn.rollback() # Rollback für diesen einen Ticker
                continue # Mache mit dem nächsten Ticker weiter
        conn.commit()

def get_gemini_sentiment_for_ticker(gemini_api_key: str, ticker: str) -> list:
    logging.info(f"Rufe Sentiment für {ticker} von Gemini ab...")
    sentiments = []
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = (
            f"Analysiere das aktuelle Sentiment für das US-Aktien-Ticker-Symbol {ticker.upper()}. "
            "Berücksichtige dabei die jüngsten Nachrichten, Marktbedingungen und Analystenmeinungen, falls allgemein bekannt. "
            "Antworte nur mit einem der folgenden Wörter: POSITIVE, NEGATIVE, oder NEUTRAL."
        )
        response = model.generate_content(prompt)
        
        sentiment_score = 0 # Default to Neutral
        raw_response_text = "Keine Antwort von Gemini."

        if response.parts:
            raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text')).strip().upper()
            logging.info(f"Gemini Sentiment Roh-Antwort für {ticker}: {raw_response_text}")
            if "POSITIVE" in raw_response_text:
                sentiment_score = 1
            elif "NEGATIVE" in raw_response_text:
                sentiment_score = -1
            elif "NEUTRAL" in raw_response_text:
                sentiment_score = 0
            else:
                logging.warning(f"Konnte Sentiment aus Gemini-Antwort für {ticker} nicht eindeutig bestimmen: {raw_response_text}")
        else:
            logging.warning(f"Keine Teile in der Gemini-Sentiment-Antwort für {ticker} gefunden.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Sentiment Prompt Feedback für {ticker}: {response.prompt_feedback}")

        sentiments.append({
            "ticker": ticker,
            "data_date": date.today(), # Aktuelles Sentiment
            "sentiment_score": sentiment_score,
            "sentiment_source": "Gemini_General_Sentiment",
            "raw_data": raw_response_text
        })
        
    except Exception as e:
        logging.error(f"Fehler bei der Gemini-Sentiment-Anfrage für {ticker}: {e}")
        # Optional: Fallback-Sentiment hinzufügen, wenn Gemini fehlschlägt
        # sentiments.append({"ticker": ticker, "data_date": date.today(), "sentiment_score": 0, "sentiment_source": "Gemini_Error_Fallback", "raw_data": str(e)})
    return sentiments


def store_sentiment_data(conn, sentiment_list: list):
    if not sentiment_list:
        logging.info("Keine Sentiment-Daten zum Speichern.")
        return

    with conn.cursor() as cur:
        for sentiment in sentiment_list:
            try:
                cur.execute("""
                    INSERT INTO sentiment_data (ticker, data_date, sentiment_score, sentiment_source, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, data_date, sentiment_source) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        raw_data = EXCLUDED.raw_data,
                        fetched_at = CURRENT_TIMESTAMP;
                """, (
                    sentiment['ticker'],
                    sentiment['data_date'],
                    sentiment['sentiment_score'],
                    sentiment['sentiment_source'],
                    sentiment['raw_data']
                ))
            except Exception as e:
                logging.error(f"Fehler beim Speichern der Sentiment-Daten für {sentiment['ticker']} am {sentiment['data_date']}: {e}")
                conn.rollback()
                continue
        conn.commit()
    logging.info(f"{len(sentiment_list)} Sentiment-Datensätze verarbeitet.")


def main_service_logic():
    logging.info("Research & FMP Service startet...")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY nicht gefunden. Überspringe Ticker-Recherche.")
        return
    # FMP_API_KEY wird nicht mehr für Sentiment benötigt, daher keine Prüfung hier,
    # es sei denn, du willst ihn für andere Zwecke im Service behalten.

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        # 1. Ticker von Gemini holen und speichern
        top_tickers = get_top_us_tickers_from_gemini(GEMINI_API_KEY, count=20)
        if top_tickers:
            store_tickers(conn, top_tickers, source="Gemini")
        else:
            logging.warning("Keine Ticker von Gemini erhalten. Verwende Fallback-Liste oder beende.")
            # Fallback: Lese Ticker, die vielleicht schon in der DB sind oder eine statische Liste
            # Für dieses Beispiel: Wenn keine Ticker von Gemini, dann keine Sentiment-Daten holen.
            return

        # 2. Sentiment-Daten von Gemini für diese Ticker holen und speichern
        all_sentiments_to_store = []
        # Lese die Ticker erneut aus der DB, um sicherzustellen, dass wir mit der aktuellen Liste arbeiten
        # (oder verwende direkt `top_tickers`, wenn keine anderen Quellen Ticker hinzufügen)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT ticker FROM target_tickers ORDER BY added_at DESC LIMIT 20;") # Holen der neuesten 20
            db_tickers_rows = cur.fetchall()

        db_tickers = [row['ticker'] for row in db_tickers_rows]
        logging.info(f"Ticker aus DB für Sentiment-Abruf: {db_tickers}")

        for ticker_symbol in db_tickers:
            sentiment_for_ticker = get_gemini_sentiment_for_ticker(GEMINI_API_KEY, ticker_symbol)
            all_sentiments_to_store.extend(sentiment_for_ticker)
            # Kurze Pause, um API-Limits nicht zu überschreiten
            time.sleep(1.5) # Pause zwischen Gemini-Aufrufen

        if all_sentiments_to_store:
            store_sentiment_data(conn, all_sentiments_to_store)

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Research & FMP Service: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")

    logging.info("Research & FMP Service hat seine Aufgabe beendet.")

if __name__ == "__main__":
    # Periodische Ausführung
    # Intervall in Sekunden (z.B. alle 24 Stunden)
    RUN_INTERVAL_SECONDS = 24 * 60 * 60

    while True:
        try:
            main_service_logic()
            logging.info(f"Research & FMP Service: Nächster Durchlauf in {RUN_INTERVAL_SECONDS / (60*60):.0f} Stunden.")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Research & FMP Service durch Benutzer beendet.")
            break
        except Exception as e:
            logging.error(f"Schwerwiegender Fehler in der Hauptschleife des Research & FMP Service: {e}. Starte in 1 Stunde neu.")
            time.sleep(60 * 60) # Warte 1 Stunde bei Fehler vor erneutem Versuch
