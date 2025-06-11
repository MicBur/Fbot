import os
import time
import alpaca_trade_api as tradeapi
import psycopg2
from dotenv import load_dotenv
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
        CREATE TABLE IF NOT EXISTS account_info (
            id SERIAL PRIMARY KEY,
            account_number VARCHAR(255) UNIQUE NOT NULL,
            buying_power NUMERIC,
            cash NUMERIC,
            portfolio_value NUMERIC,
            equity NUMERIC,
            currency VARCHAR(10),
            status VARCHAR(50),
            last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            asset_id VARCHAR(255) PRIMARY KEY,
            symbol VARCHAR(50) UNIQUE NOT NULL,
            qty NUMERIC NOT NULL,
            avg_entry_price NUMERIC NOT NULL,
            current_price NUMERIC,
            market_value NUMERIC,
            cost_basis NUMERIC,
            unrealized_pl NUMERIC,
            unrealized_plpc NUMERIC,
            change_today NUMERIC,
            lastday_price NUMERIC,
            exchange VARCHAR(50),
            asset_class VARCHAR(50),
            side VARCHAR(10),
            last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
    logging.info("Datenbank-Tabellen sichergestellt.")

def main():
    logging.info("Alpaca Service startet...")

    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

    if not all([ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        logging.error("Fehler: Nicht alle notwendigen Umgebungsvariablen für Alpaca Service sind gesetzt.")
        return

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    logging.info(f"Alpaca Service: Paper-Modus ist {'aktiv' if ALPACA_PAPER else 'inaktiv'}.")

    conn = None
    try:
        conn = get_db_connection()
        create_tables(conn)

        # Account-Informationen abrufen und speichern
        account = api.get_account()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO account_info (account_number, buying_power, cash, portfolio_value, equity, currency, status, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (account_number) DO UPDATE SET
                    buying_power = EXCLUDED.buying_power,
                    cash = EXCLUDED.cash,
                    portfolio_value = EXCLUDED.portfolio_value,
                    equity = EXCLUDED.equity,
                    currency = EXCLUDED.currency,
                    status = EXCLUDED.status,
                    last_updated = CURRENT_TIMESTAMP;
            """, (account.account_number, account.buying_power, account.cash, account.portfolio_value, account.equity, account.currency, account.status))
            conn.commit()
        logging.info(f"Account-Informationen für {account.account_number} gespeichert.")

        # Bestehende Positionen löschen, um veraltete Einträge zu entfernen (oder spezifischer aktualisieren)
        # Einfacher Ansatz: Alle alten Positionen löschen und neu einfügen.
        # Ein besserer Ansatz wäre, nur nicht mehr vorhandene Positionen zu löschen.
        # Für den Moment leeren wir die Tabelle und fügen alle aktuellen Positionen ein.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM positions;") # Vorsicht: Löscht alle Positionen vor jedem Update
            conn.commit()
        logging.info("Alte Positionen aus der Datenbank entfernt.")

        # Positionen abrufen und speichern
        positions = api.list_positions()
        if positions:
            with conn.cursor() as cur:
                for p in positions:
                    cur.execute("""
                        INSERT INTO positions (asset_id, symbol, qty, avg_entry_price, current_price, market_value, cost_basis, unrealized_pl, unrealized_plpc, change_today, lastday_price, exchange, asset_class, side, last_updated)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (asset_id) DO UPDATE SET
                            symbol = EXCLUDED.symbol,
                            qty = EXCLUDED.qty,
                            avg_entry_price = EXCLUDED.avg_entry_price,
                            current_price = EXCLUDED.current_price,
                            market_value = EXCLUDED.market_value,
                            cost_basis = EXCLUDED.cost_basis,
                            unrealized_pl = EXCLUDED.unrealized_pl,
                            unrealized_plpc = EXCLUDED.unrealized_plpc,
                            change_today = EXCLUDED.change_today,
                            lastday_price = EXCLUDED.lastday_price,
                            exchange = EXCLUDED.exchange,
                            asset_class = EXCLUDED.asset_class,
                            side = EXCLUDED.side,
                            last_updated = CURRENT_TIMESTAMP;
                    """, (p.asset_id, p.symbol, p.qty, p.avg_entry_price, p.current_price, p.market_value, p.cost_basis, p.unrealized_pl, p.unrealized_plpc, p.change_today, p.lastday_price, p.exchange, p.asset_class, p.side))
                conn.commit()
            logging.info(f"{len(positions)} Position(en) gespeichert/aktualisiert.")
        else:
            logging.info("Keine offenen Positionen gefunden.")

    except Exception as e:
        logging.error(f"Fehler im Alpaca Service: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")
    logging.info("Alpaca Service hat seine Aufgabe beendet.")

if __name__ == "__main__":
    # Periodische Ausführung
    # Intervall in Sekunden (z.B. alle 15 Minuten)
    RUN_INTERVAL_SECONDS = 15 * 60 

    while True:
        try:
            main()
            logging.info(f"Alpaca Service: Nächster Durchlauf in {RUN_INTERVAL_SECONDS / 60:.0f} Minuten.")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Alpaca Service durch Benutzer beendet.")
            break
        except Exception as e:
            logging.error(f"Schwerwiegender Fehler in der Hauptschleife des Alpaca Service: {e}. Starte in 5 Minuten neu.")
            time.sleep(5 * 60) # Warte 5 Minuten bei Fehler vor erneutem Versuch