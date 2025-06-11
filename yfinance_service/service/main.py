import os
import time
import yfinance as yf
import psycopg2
from dotenv import load_dotenv
import psycopg2.extras # Für DictCursor
import logging

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
        CREATE TABLE IF NOT EXISTS stock_data_ohlc (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            adj_close NUMERIC,
            volume BIGINT,
            last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker, date)
        );
        """)
        conn.commit()
    logging.info("Tabelle 'stock_data_ohlc' sichergestellt.")

def main():
    logging.info("Yahoo Finance Service startet...")

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        tickers_to_fetch = []
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Hole alle Ticker aus der target_tickers Tabelle
            # Optional: Filtere nach Quelle oder füge ein Limit hinzu
            cur.execute("SELECT ticker FROM target_tickers ORDER BY ticker ASC;")
            rows = cur.fetchall()
            tickers_to_fetch = [row['ticker'] for row in rows]

        if not tickers_to_fetch:
            logging.info("Keine Ticker in der 'target_tickers'-Tabelle gefunden. Überspringe Datenabruf.")
            return
        logging.info(f"Ticker aus 'target_tickers' für den Abruf geladen: {tickers_to_fetch}")

        for ticker_symbol in tickers_to_fetch:
            logging.info(f"Rufe historische Daten für {ticker_symbol} ab...")
            try:
                data_found = False
                current_ticker_to_try = ticker_symbol
                
                # Erster Versuch mit dem Original-Ticker
                stock = yf.Ticker(current_ticker_to_try)
                hist_data = stock.history(period="5y", interval="1d")

                if not hist_data.empty:
                    data_found = True
                # Wenn keine Daten gefunden wurden UND der Ticker einen Punkt enthält, versuche es mit Bindestrich
                elif "." in current_ticker_to_try:
                    alternative_ticker = current_ticker_to_try.replace(".", "-")
                    logging.info(f"Keine Daten für {current_ticker_to_try} gefunden. Versuche Alternative: {alternative_ticker}")
                    stock = yf.Ticker(alternative_ticker)
                    hist_data = stock.history(period="5y", interval="1d")
                    if not hist_data.empty:
                        data_found = True
                        current_ticker_to_try = alternative_ticker # Wichtig für das Speichern unter dem korrekten Ticker

                if not data_found:
                    logging.warning(f"Keine Daten für {ticker_symbol} gefunden.")
                    continue

                with conn.cursor() as cur:
                    for index, row in hist_data.iterrows():
                        # Konvertiere Timestamp zu Date-Objekt
                        date_obj = index.date()
                        # Sicherstellen, dass 'Adj Close' vorhanden ist, sonst 'Close' verwenden
                        adj_close_val = row.get('Adj Close') 
                        if adj_close_val is None:
                            adj_close_val = row.get('Close') # Fallback
                        if adj_close_val is None:
                            logging.warning(f"Weder 'Adj Close' noch 'Close' für {ticker_symbol} am {date_obj} gefunden. Überspringe Adj Close.")
                        cur.execute("""
                            INSERT INTO stock_data_ohlc (ticker, date, open, high, low, close, adj_close, volume, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                            ON CONFLICT (ticker, date) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                adj_close = EXCLUDED.adj_close,
                                volume = EXCLUDED.volume,
                                last_updated = CURRENT_TIMESTAMP;
                        """, (
                            current_ticker_to_try, # Verwende den Ticker, mit dem Daten gefunden wurden
                            date_obj,
                            float(row['Open']) if row['Open'] is not None else None,
                            float(row['High']) if row['High'] is not None else None,
                            float(row['Low']) if row['Low'] is not None else None,
                            float(row['Close']) if row['Close'] is not None else None,
                            float(adj_close_val) if adj_close_val is not None else None,
                            int(row['Volume']) if row['Volume'] is not None else None
                        ))
                    conn.commit()
                logging.info(f"{len(hist_data)} Datensätze für {ticker_symbol} gespeichert/aktualisiert.")
            except Exception as e:
                logging.error(f"Fehler beim Verarbeiten von {ticker_symbol}: {e}")
                if conn:
                    conn.rollback() # Rollback für den aktuellen Ticker

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Yahoo Finance Service: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")
    logging.info("Yahoo Finance Service hat seine Aufgabe beendet.")

if __name__ == "__main__":
    # Periodische Ausführung
    # Intervall in Sekunden (z.B. alle 24 Stunden)
    RUN_INTERVAL_SECONDS = 24 * 60 * 60

    while True:
        try:
            main()
            logging.info(f"Yahoo Finance Service: Nächster Durchlauf in {RUN_INTERVAL_SECONDS / (60*60):.0f} Stunden.")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Yahoo Finance Service durch Benutzer beendet.")
            break
        except Exception as e:
            logging.error(f"Schwerwiegender Fehler in der Hauptschleife des Yahoo Finance Service: {e}. Starte in 1 Stunde neu.")
            time.sleep(60 * 60) # Warte 1 Stunde bei Fehler vor erneutem Versuch