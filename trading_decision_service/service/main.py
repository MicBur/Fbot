import os
import time
import psycopg2
import psycopg2.extras
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import logging
from decimal import Decimal # Für präzise Finanzberechnungen
from typing import Optional # Hinzugefügt für Type Hinting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Konfiguration ---
RISK_BUFFER_CURRENCY = Decimal(os.getenv("RISK_BUFFER_CURRENCY", "10000")) # z.B. 10000 EUR/USD
MAX_PORTFOLIO_SIZE = int(os.getenv("MAX_PORTFOLIO_SIZE", "20")) # Maximale Anzahl verschiedener Aktien
ORDER_TYPE = os.getenv("ORDER_TYPE", "market") # market, limit, etc.
ORDER_TIME_IN_FORCE = os.getenv("ORDER_TIME_IN_FORCE", "day") # day, gtc, etc.
# Prozentsatz des *verfügbaren Handelskapitals*, der für einen *neuen* Trade verwendet werden soll.
CAPITAL_PER_NEW_TRADE_PERCENTAGE = Decimal(os.getenv("CAPITAL_PER_NEW_TRADE_PERCENTAGE", "0.05")) # 5%
PREDICTION_DEVIATION_THRESHOLD = Decimal(os.getenv("PREDICTION_DEVIATION_THRESHOLD", "0.10")) # 10% Abweichung
# Prozentsatz des verfügbaren Kapitals (nach Puffer), der pro Trade eingesetzt werden kann
# oder ein fester Betrag. Für den Anfang halten wir es einfach.

def get_db_connection():
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432")
        logging.info("TradingDecisionService: Datenbankverbindung erfolgreich hergestellt.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"TradingDecisionService: Fehler beim Herstellen der DB-Verbindung: {e}")
        raise

def create_db_tables_if_not_exist(conn):
    """Erstellt die notwendigen Tabellen, falls sie noch nicht existieren."""
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS retraining_flags (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) UNIQUE NOT NULL, -- Ein Flag pro Ticker
            reason TEXT,
            status VARCHAR(50) DEFAULT 'pending', -- pending, in_progress, completed, error
            flagged_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            last_checked_by_trainer TIMESTAMPTZ
        );""")
        conn.commit()
    logging.info("Tabelle 'retraining_flags' sichergestellt.")

def get_alpaca_api_client():
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logging.error("Alpaca API Key oder Secret Key nicht konfiguriert.")
        return None
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')
    logging.info(f"Alpaca API Client initialisiert. Paper-Modus: {ALPACA_PAPER}")
    return api

def get_latest_predictions(conn):
    """Holt die neuesten Vorhersagen für jeden Ticker."""
    predictions = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Diese Abfrage holt die neueste Vorhersage pro Ticker, basierend auf generated_at
            # und stellt sicher, dass das Vorhersagedatum in der Zukunft liegt.
            cur.execute("""
                SELECT p.ticker, p.prediction_date, p.predicted_value, p.model_name, p.generated_at
                FROM model_predictions p
                INNER JOIN (
                    SELECT ticker, MAX(generated_at) as max_generated_at
                    FROM model_predictions
                    WHERE prediction_date >= CURRENT_DATE -- Nur Vorhersagen für heute oder die Zukunft
                    GROUP BY ticker
                ) pm ON p.ticker = pm.ticker AND p.generated_at = pm.max_generated_at
                ORDER BY p.ticker;
            """)
            rows = cur.fetchall()
            for row in rows:
                predictions[row['ticker']] = row
        logging.info(f"{len(predictions)} aktuelle Vorhersagen geladen.")
    except Exception as e:
        logging.error(f"Fehler beim Laden der Vorhersagen: {e}")
    return predictions

def get_account_info(conn) -> Optional[psycopg2.extras.DictRow]:
    """Holt die neuesten Kontoinformationen."""
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM account_info ORDER BY last_updated DESC LIMIT 1;")
            account = cur.fetchone()
            if account:
                logging.info(f"Kontoinformationen geladen: Cash {account['cash']}, Equity {account['equity']}")
            else:
                logging.warning("Keine Kontoinformationen in der DB gefunden.")
            return account
    except Exception as e:
        logging.error(f"Fehler beim Laden der Kontoinformationen: {e}")
        return None

def get_current_positions(conn) -> dict:
    """Holt die aktuellen Positionen aus der Datenbank."""
    positions = {}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT symbol, qty, avg_entry_price, current_price, market_value FROM positions;")
            rows = cur.fetchall()
            for row in rows:
                positions[row['symbol']] = row
        logging.info(f"{len(positions)} aktuelle Positionen aus DB geladen.")
    except Exception as e:
        logging.error(f"Fehler beim Laden der Positionen aus DB: {e}")
    return positions

def check_prediction_deviations_and_flag(conn, current_positions_db: dict, alpaca_api: tradeapi.REST):
    logging.info("Prüfe Vorhersageabweichungen für gehaltene Positionen...")
    date_str = time.strftime("%Y-%m-%d")
    year, month, day = map(int, date_str.split('-'))
    today_db_date = psycopg2.Date(year, month, day) # Aktuelles Datum als Date-Objekt für DB-Abfragen

    for ticker, position_data in current_positions_db.items():
        if not isinstance(position_data, psycopg2.extras.DictRow):
            logging.warning(f"Position data for {ticker} is not in the expected DictRow format. Skipping deviation check for this ticker.")
            continue
        try:
            # Hole die letzte Vorhersage für heute (oder gestern, falls heute noch keine generiert wurde)
            # für diesen Ticker.
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT predicted_value
                    FROM model_predictions
                    WHERE ticker = %s AND prediction_date <= %s
                    ORDER BY prediction_date DESC, generated_at DESC
                    LIMIT 1;
                """, (ticker, today_db_date))
                latest_relevant_prediction = cur.fetchone()

            if not latest_relevant_prediction:
                logging.debug(f"Keine relevante vergangene/heutige Vorhersage für gehaltenen Ticker {ticker} gefunden.")
                continue

            predicted_price_for_today = Decimal(latest_relevant_prediction['predicted_value'])
            
            # Hole aktuellen Marktpreis
            current_market_price = Decimal(position_data.get('current_price', 0))
            if not current_market_price or current_market_price <= 0:
                 # Versuche, von Alpaca zu holen, falls DB-Wert veraltet/0 ist
                try:
                    snapshot = alpaca_api.get_snapshot(ticker)
                    if snapshot and snapshot.latest_trade:
                        current_market_price = Decimal(snapshot.latest_trade.p)
                    elif snapshot and snapshot.latest_quote:
                        current_market_price = (Decimal(snapshot.latest_quote.ap) + Decimal(snapshot.latest_quote.bp)) / 2
                except Exception as e_snap:
                    logging.warning(f"Konnte aktuellen Preis für {ticker} (Deviation Check) nicht von Alpaca abrufen: {e_snap}")
                    continue
            
            if not current_market_price or current_market_price <= 0:
                logging.warning(f"Ungültiger aktueller Marktpreis {current_market_price} für {ticker} (Deviation Check).")
                continue

            deviation = abs(current_market_price - predicted_price_for_today) / current_market_price
            logging.debug(f"Deviation Check für {ticker}: Aktuell={current_market_price:.2f}, Vorhergesagt (für heute/kurzfristig)={predicted_price_for_today:.2f}, Abweichung={deviation:.2%}")

            if deviation > PREDICTION_DEVIATION_THRESHOLD:
                logging.info(f"Starke Abweichung ({deviation:.2%}) für {ticker} festgestellt. Setze Retraining-Flag.")
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO retraining_flags (ticker, reason, status, flagged_at)
                        VALUES (%s, %s, 'pending', CURRENT_TIMESTAMP)
                        ON CONFLICT (ticker) DO UPDATE SET
                            reason = EXCLUDED.reason,
                            status = 'pending', 
                            flagged_at = CURRENT_TIMESTAMP;
                    """, (ticker, f"Deviation: {deviation:.2%}, Actual: {current_market_price}, Predicted: {predicted_price_for_today}"))
                    conn.commit()
            else:
                # Optional: Flag zurücksetzen, wenn Abweichung wieder im Rahmen ist?
                # Oder nur setzen und der model_training_service setzt es nach Training zurück.
                pass

        except Exception as e:
            logging.error(f"Fehler bei der Abweichungsprüfung für {ticker}: {e}", exc_info=True)

def make_trading_decision_and_execute(conn, api: tradeapi.REST, predictions: dict, account_info: dict, current_positions_db: dict):
    if not api or not account_info:
        logging.error("API-Client oder Kontoinformationen nicht verfügbar. Keine Trades möglich.")
        return

    cash_available_for_trading = Decimal(account_info.get('cash', 0)) - RISK_BUFFER_CURRENCY
    if cash_available_for_trading <= 0:
        logging.info("Nicht genügend Cash für Trading nach Abzug des Puffers verfügbar.")
        # Hier könnte auch eine Logik zum Verkaufen von Positionen implementiert werden,
        # falls Vorhersagen negativ sind, unabhängig vom Cash.
        # return # Für den Moment beenden wir hier, wenn kein Cash für Käufe da ist.

    logging.info(f"Verfügbares Kapital für Trading (nach Puffer): {cash_available_for_trading:.2f} {account_info.get('currency')}")

    # --- Hier beginnt die eigentliche Handelslogik ---
    # Diese ist stark vereinfacht und muss ausgebaut werden!

    for ticker, pred_data in predictions.items():
        predicted_price = Decimal(pred_data['predicted_value'])
        prediction_date = pred_data['prediction_date']
        
        # Hole aktuellen Preis für den Ticker (wichtig für die Entscheidung)
        try:
            # Verwende Alpaca, um den aktuellen Preis zu bekommen (oder Snapshot)
            # Für Aktien, die nicht im Portfolio sind, ist ein direkter Quote-Abruf nötig.
            # Für Aktien im Portfolio ist der current_price in current_positions_db eine gute Annäherung.
            
            current_market_price = None
            if ticker in current_positions_db:
                current_market_price = Decimal(current_positions_db[ticker].get('current_price', 0))
            
            if not current_market_price: # Fallback oder für neue Ticker
                # Dies kann teuer sein, wenn für viele Ticker gemacht.
                # Besser: Batch-Abfrage oder Snapshots, falls von Alpaca API unterstützt.
                # Für den Moment: Einfacher Abruf, wenn nicht in Positionen.
                # Vorsicht: `get_latest_trade` ist für einen einzelnen Trade, `get_latest_quote` für Bid/Ask.
                # `get_snapshot` ist oft besser.
                try:
                    snapshot = api.get_snapshot(ticker)
                    if snapshot and snapshot.latest_trade:
                        current_market_price = Decimal(snapshot.latest_trade.p)
                    elif snapshot and snapshot.latest_quote: # Fallback auf Mid-Preis
                        current_market_price = (Decimal(snapshot.latest_quote.ap) + Decimal(snapshot.latest_quote.bp)) / 2
                    else:
                        logging.warning(f"Konnte aktuellen Preis für {ticker} nicht von Alpaca abrufen.")
                        continue
                except Exception as e:
                    logging.error(f"Fehler beim Abrufen des aktuellen Preises für {ticker} von Alpaca: {e}")
                    continue
            
            if not current_market_price or current_market_price <= 0:
                logging.warning(f"Ungültiger aktueller Marktpreis {current_market_price} für {ticker}. Überspringe.")
                continue

            logging.info(f"Ticker: {ticker}, Vorhersage ({prediction_date}): {predicted_price:.2f}, Aktuell: {current_market_price:.2f}")

            # Beispielhafte Kauf-Logik (sehr einfach)
            # Kaufe, wenn Vorhersage > aktueller Preis + x% und wir Cash haben und Portfolio nicht voll ist
            profit_target_percentage = Decimal("0.02") # 2% erwarteter Gewinn als Schwelle
            
            if predicted_price > current_market_price * (1 + profit_target_percentage):
                if len(current_positions_db) < MAX_PORTFOLIO_SIZE or ticker in current_positions_db:
                    # Kapital für diesen spezifischen Trade
                    # Wenn es eine neue Position ist und das Portfolio noch nicht voll ist:
                    is_new_position = ticker not in current_positions_db
                    
                    # Für neue Positionen: Verwende einen Prozentsatz des gesamten verfügbaren Handelskapitals
                    # Für bestehende Positionen: Hier könnte eine andere Logik gelten (z.B. nicht weiter aufstocken oder nur bis zu einem Max-Wert)
                    # Für den Moment behandeln wir das Aufstocken bestehender Positionen nicht gesondert mit Kapitalallokation.
                    # Wir konzentrieren uns auf neue Käufe.
                    
                    if is_new_position and len(current_positions_db) >= MAX_PORTFOLIO_SIZE:
                        logging.info(f"Portfolio ist voll ({len(current_positions_db)}/{MAX_PORTFOLIO_SIZE}). Kein Kauf von neuem Ticker {ticker}.")
                        continue

                    # Berechne die zu investierende Summe für diesen Trade
                    # Wenn es eine neue Position ist, nimm den Prozentsatz des gesamten verfügbaren Kapitals
                    # Wenn es eine bestehende Position ist, könnten wir z.B. eine feste kleine Menge kaufen oder nicht aufstocken.
                    # Hier vereinfacht: Wir kaufen nur, wenn wir Cash haben.
                    amount_to_invest_this_trade = cash_available_for_trading * CAPITAL_PER_NEW_TRADE_PERCENTAGE
                    
                    if current_market_price > 0 and amount_to_invest_this_trade > current_market_price: # Mindestens 1 Aktie muss kaufbar sein
                        qty_to_buy = int(amount_to_invest_this_trade // current_market_price) # Nur ganze Aktien
                        
                        if qty_to_buy > 0:
                            logging.info(f"KAUF-Signal für {ticker}: Vorhersage {predicted_price:.2f} > Aktuell {current_market_price:.2f} + Schwelle.")
                            try:
                                api.submit_order(
                                    symbol=ticker,
                                    qty=qty_to_buy,
                                    side='buy',
                                    type=ORDER_TYPE,
                                    time_in_force=ORDER_TIME_IN_FORCE
                                )
                                logging.info(f"Kauforder für {qty_to_buy} Stück von {ticker} platziert.")
                                cash_available_for_trading -= (current_market_price * qty_to_buy) # Reduziere verfügbares Cash für diese Iteration
                            except Exception as e:
                                logging.error(f"Fehler beim Platzieren der Kauforder für {ticker}: {e}")
                        else:
                            logging.info(f"Berechnete Menge für {ticker} ist 0 (Investitionssumme: {amount_to_invest_this_trade:.2f}, Preis: {current_market_price:.2f}).")
                    else:
                        logging.info(f"Nicht genügend Kapital für einen Trade in {ticker} (Verfügbar für Trade: {amount_to_invest_this_trade:.2f}, Preis: {current_market_price:.2f}) oder Marktpreis ist 0.")
                        
            # Beispielhafte Verkaufs-Logik (sehr einfach)
            # Verkaufe, wenn Vorhersage < aktueller Preis - x% ODER wenn ein Stop-Loss erreicht wird (nicht implementiert)
            elif ticker in current_positions_db and predicted_price < current_market_price * (1 - profit_target_percentage):
                position_data = current_positions_db[ticker]
                qty_held = Decimal(position_data.get('qty', 0))
                if qty_held > 0:
                    logging.info(f"VERKAUF-Signal für {ticker}: Vorhersage {predicted_price:.2f} < Aktuell {current_market_price:.2f} - Schwelle.")
                    try:
                        api.submit_order(
                            symbol=ticker,
                            qty=qty_held, # Verkaufe die gesamte Position
                            side='sell',
                            type=ORDER_TYPE,
                            time_in_force=ORDER_TIME_IN_FORCE
                        )
                        logging.info(f"Verkaufsorder für {qty_held} Stück von {ticker} platziert.")
                    except Exception as e:
                        logging.error(f"Fehler beim Platzieren der Verkaufsorder für {ticker}: {e}")
            else:
                logging.info(f"HALTEN-Signal für {ticker}.")
                
        except Exception as e:
            logging.error(f"Fehler bei der Entscheidungsfindung für Ticker {ticker}: {e}", exc_info=True)
            continue # Zum nächsten Ticker

    logging.info("Handelsentscheidungen abgeschlossen.")


def main_service_loop():
    logging.info("Trading Decision Service Loop startet...")
    conn = None
    api = None
    try:
        # Stelle sicher, dass die Tabellen existieren (insbesondere retraining_flags)
        conn = get_db_connection()
        api = get_alpaca_api_client()

        if not conn or not api:
            logging.error("Konnte DB-Verbindung oder Alpaca API nicht initialisieren. Beende Loop-Iteration.")
            return

        create_db_tables_if_not_exist(conn) # Stellt sicher, dass retraining_flags existiert
        predictions = get_latest_predictions(conn)
        account_info = get_account_info(conn)
        current_positions_db = get_current_positions(conn) # Positionen aus unserer DB

        if not predictions:
            logging.info("Keine aktuellen Vorhersagen gefunden. Überspringe Handelslogik.")
            return
        
        if not account_info:
            logging.warning("Keine Kontoinformationen. Überspringe Handelslogik.")
            return

        # Prüfe auf Abweichungen und setze ggf. Retraining-Flags
        check_prediction_deviations_and_flag(conn, current_positions_db, api)

        make_trading_decision_and_execute(conn, api, predictions, account_info, current_positions_db)

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Trading Decision Service Loop: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("TradingDecisionService: Datenbankverbindung geschlossen.")
    logging.info("Trading Decision Service Loop beendet.")


if __name__ == "__main__":
    RUN_INTERVAL_SECONDS = int(os.getenv("TRADING_DECISION_INTERVAL_SECONDS", 15 * 60)) # z.B. alle 15 Minuten

    while True:
        # Hier könnte eine Logik eingefügt werden, um nur während der Handelszeiten zu laufen
        market_open = False
        temp_api_for_clock = get_alpaca_api_client()
        if temp_api_for_clock:
            try:
                clock = temp_api_for_clock.get_clock()
                if clock.is_open:
                    market_open = True
                    logging.info("Markt ist geöffnet. Starte Handelslogik.")
                    main_service_loop()
                else:
                    logging.info(f"Markt ist geschlossen. Nächste Öffnung: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S%z') if clock.next_open else 'Unbekannt'}")
            except Exception as e:
                logging.error(f"Fehler beim Überprüfen des Marktstatus: {e}")
            finally:
                # Explizit schließen, wenn die API-Instanz nur temporär ist (nicht unbedingt nötig bei Alpaca SDK, aber gute Praxis)
                # Das SDK handhabt Sessions intern, aber um sicherzugehen.
                pass # temp_api_for_clock.close() # falls eine close Methode existiert
        else:
            logging.error("Kann Alpaca API nicht für Marktstatusprüfung initialisieren. Überspringe Handelslogik.")

        try:
            logging.info(f"Trading Decision Service: Nächster Durchlauf in {RUN_INTERVAL_SECONDS / 60:.0f} Minuten.")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Trading Decision Service durch Benutzer beendet.")
            break
        except Exception as e: # Fängt andere Fehler in der Hauptschleife ab (z.B. time.sleep unterbrochen)
            logging.error(f"Fehler in der Hauptschleife des Trading Decision Service (außerhalb der Loop-Logik): {e}", exc_info=True)
            logging.info("Warte 5 Minuten vor dem nächsten Versuch.")
            time.sleep(5 * 60)
