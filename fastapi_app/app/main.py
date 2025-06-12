from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles # Auskommentiert, falls nicht direkt genutzt
from fastapi.templating import Jinja2Templates
import os
import psycopg2
import redis
import psycopg2.extras # Für DictCursor
from dotenv import load_dotenv
import logging
from typing import Optional # Hinzugefügt für Typ-Hinweise
from pydantic import BaseModel # Für Request Body
from datetime import datetime, timedelta # Für potential_buys
from zoneinfo import ZoneInfo # Für Zeitzonen
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Lädt Variablen aus der .env Datei, falls vorhanden und nicht durch Docker Compose gesetzt

app = FastAPI()

# Mount static files (z.B. CSS, JS) - Erstelle ein 'static' Verzeichnis in 'fastapi_app'
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Datenbank-Setup ---
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True) # pool_pre_ping hinzugefügt
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Templates ---
templates_dir = os.path.join(os.path.dirname(__file__), "templates") # Korrigierter Pfad
templates = Jinja2Templates(directory=templates_dir)

# --- Redis Setup ---
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0, decode_responses=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    db_status = "PostgreSQL: nicht verbunden"
    redis_status = "Redis: nicht verbunden"
    
    try:
        # Test DB connection
        db.execute(text("SELECT 1"))
        db_status = "PostgreSQL: erfolgreich verbunden"
    except Exception as e:
        db_status = f"PostgreSQL: Verbindungsfehler - {e}"
            
    try:
        redis_client.ping()
        redis_status = "Redis: erfolgreich verbunden"
    except Exception as e:
        redis_status = f"Redis: Verbindungsfehler - {e}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "db_status": db_status,
        "redis_status": redis_status
    })

@app.get("/portfolio", response_class=HTMLResponse)
async def get_portfolio(request: Request, db: Session = Depends(get_db)):
    account_info = None
    positions = []
    error_message = None
    settings_data = {}
    german_time = None
    pie_chart_data = None
    total_weekly_profit_loss = None

    try:
        now_utc = datetime.now(ZoneInfo("UTC"))
        german_time = now_utc.astimezone(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y %H:%M:%S %Z")
    except Exception as e:
        logging.error(f"Fehler beim Ermitteln der deutschen Zeit: {e}")

    try:
        settings_result = db.execute(text("SELECT setting_key, setting_value FROM bot_settings WHERE setting_key = 'portfolio_refresh_interval';")).fetchone()
        if settings_result:
            settings_data[settings_result.setting_key] = {"value": settings_result.setting_value}
    except Exception as e:
        logging.error(f"Fehler beim Laden der Portfolio-Refresh-Einstellung: {e}")

    try:
        account_info_result = db.execute(text("SELECT * FROM account_info ORDER BY last_updated DESC LIMIT 1;")).fetchone()
        if account_info_result:
            # Konvertiere RowProxy zu einem Dict-ähnlichen Objekt, falls nötig, oder greife direkt auf Spalten zu
            account_info = dict(account_info_result._mapping)

        positions_result = db.execute(text("SELECT * FROM positions ORDER BY symbol;")).fetchall()
        positions = [dict(row._mapping) for row in positions_result]

        if positions:
            # Daten für Tortendiagramm (Marktwertverteilung)
            pie_labels = [pos['symbol'] for pos in positions]
            pie_values = [float(pos['market_value']) for pos in positions]
            pie_chart_data = {"labels": pie_labels, "values": pie_values}

            # Berechnung des Gewinns/Verlusts der laufenden Woche
            today = datetime.now(ZoneInfo("Europe/Berlin")).date() # Deutsche Zeit für Wochenstart
            start_of_week_date = today - timedelta(days=today.weekday()) # Montag dieser Woche
            
            weekly_pls = []
            for pos in positions:
                try:
                    # Hole den letzten Schlusskurs am oder vor dem Wochenstart
                    price_at_week_start_query = text("""
                        SELECT close FROM stock_data_ohlc
                        WHERE ticker = :ticker AND date <= :start_date
                        ORDER BY date DESC LIMIT 1;
                    """)
                    price_start_row = db.execute(price_at_week_start_query, 
                                                 {"ticker": pos['symbol'], "start_date": start_of_week_date}).fetchone()
                    
                    if price_start_row and price_start_row.close is not None:
                        value_at_week_start = float(price_start_row.close) * float(pos['qty'])
                        current_value = float(pos['market_value'])
                        weekly_pl_for_stock = current_value - value_at_week_start
                        weekly_pls.append(weekly_pl_for_stock)
                    else:
                        # Fallback oder Log, falls kein Startpreis gefunden wurde
                        logging.warning(f"Kein Startpreis für {pos['symbol']} zu Wochenbeginn gefunden. Wochen-G/V kann nicht berechnet werden.")
                except Exception as e_weekly_pl:
                    logging.error(f"Fehler bei Berechnung Wochen-G/V für {pos['symbol']}: {e_weekly_pl}")
            
            if weekly_pls:
                total_weekly_profit_loss = sum(weekly_pls)

    except Exception as e:
        error_message = f"Fehler beim Abrufen der Portfoliodaten: {e}"
        logging.error(error_message)
    
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "account_info": account_info,
        "positions": positions,
        "error_message": error_message,
        "settings_data": settings_data,
        "german_time": german_time,
        "pie_chart_data": pie_chart_data,
        "total_weekly_profit_loss": total_weekly_profit_loss
    })

@app.get("/tickers", response_class=HTMLResponse)
async def get_tickers_page(request: Request, db: Session = Depends(get_db)):
    tickers = []
    try:
        result = db.execute(text("SELECT ticker, name, source, added_at FROM target_tickers ORDER BY ticker ASC;")).fetchall()
        tickers = [dict(row._mapping) for row in result]
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Ticker-Liste: {e}")
        # Optional: eine Fehlermeldung an das Template übergeben
    return templates.TemplateResponse("tickers.html", {"request": request, "tickers": tickers})

@app.get("/tickers/{ticker_symbol}", response_class=HTMLResponse)
async def read_ticker_detail(request: Request, ticker_symbol: str):
    return templates.TemplateResponse("ticker_detail.html", {"request": request, "ticker_symbol": ticker_symbol.upper()})

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request, db: Session = Depends(get_db)):
    settings = {}
    try:
        result = db.execute(text("SELECT setting_key, setting_value, description FROM bot_settings;")).fetchall()
        for row_proxy in result:
            row = dict(row_proxy._mapping)
            settings[row['setting_key']] = {"value": row['setting_value'], "description": row['description']}
    except Exception as e:
        logging.error(f"Fehler beim Laden der Bot-Einstellungen: {e}")
        # Seite trotzdem laden, ggf. mit Fehlermeldung oder Standardwerten im JS
    return templates.TemplateResponse("settings.html", {"request": request, "settings_data": settings})


@app.get("/api/v1/chart_data/{ticker_symbol}")
async def get_chart_data(ticker_symbol: str, history_days: Optional[int] = None, db: Session = Depends(get_db)):
    try:
        # Standardwert für history_days, falls nicht übergeben
        effective_history_days = history_days if history_days is not None else 14

        historical_data_query = text(f"""
            SELECT date, close 
            FROM stock_data_ohlc
            WHERE ticker = :ticker 
              AND date >= (SELECT MAX(date) FROM stock_data_ohlc WHERE ticker = :ticker) - INTERVAL '{effective_history_days} days'
              AND date < (SELECT MAX(date) FROM stock_data_ohlc WHERE ticker = :ticker) + INTERVAL '1 day'
            ORDER BY date ASC;
        """)
        historical_result = db.execute(historical_data_query, {"ticker": ticker_symbol}).fetchall()
        historical_prices = [{"date": str(row.date), "price": float(row.close)} for row in historical_result]

        # Hole die letzten N Vorhersageläufe
        # Ein "Lauf" wird durch generated_at identifiziert
        # Wir holen z.B. die letzten 3 Läufe
        prediction_runs_query = text(f"""
            WITH RankedPredictionRuns AS (
                SELECT
                    ticker,
                    prediction_date,
                    predicted_value,
                    generated_at,
                    DENSE_RANK() OVER (PARTITION BY ticker ORDER BY generated_at DESC) as run_rank
                FROM model_predictions
                WHERE ticker = :ticker
                  AND prediction_date >= CURRENT_DATE 
                  AND prediction_date < CURRENT_DATE + INTERVAL '8 days'
            )
            SELECT
                ticker,
                prediction_date,
                predicted_value,
                generated_at
            FROM RankedPredictionRuns
            WHERE run_rank <= 3 -- Anzahl der anzuzeigenden Vorhersageläufe
            ORDER BY generated_at DESC, prediction_date ASC;
        """)
        prediction_result = db.execute(prediction_runs_query, {"ticker": ticker_symbol}).fetchall()
        
        prediction_runs_data = {}
        for row in prediction_result:
            run_ts = str(row.generated_at)
            if run_ts not in prediction_runs_data:
                prediction_runs_data[run_ts] = []
            prediction_runs_data[run_ts].append({"date": str(row.prediction_date), "price": float(row.predicted_value)})

        # Umwandeln in eine Liste von Objekten für das Frontend
        prediction_runs_list = [{"generated_at": ts, "predictions": preds} for ts, preds in prediction_runs_data.items()]

        return {"ticker": ticker_symbol, "historical": historical_prices, "prediction_runs": prediction_runs_list}
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Chart-Daten für {ticker_symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden der Chart-Daten für {ticker_symbol}")

@app.get("/api/v1/sentiment_data/{ticker_symbol}")
async def get_sentiment_data(ticker_symbol: str, db: Session = Depends(get_db)):
    try:
        # Sentiment-Daten (z.B. letzte 90 Tage)
        sentiment_data_query = text("""
            SELECT data_date, sentiment_score, sentiment_source
            FROM sentiment_data
            WHERE ticker = :ticker
            ORDER BY data_date DESC
            LIMIT 90;
        """)
        sentiment_result = db.execute(sentiment_data_query, {"ticker": ticker_symbol}).fetchall()
        sentiments = [{"date": str(row.data_date), "score": float(row.sentiment_score), "source": str(row.sentiment_source)} for row in sentiment_result]
        sentiments.reverse() # Für chronologische Reihenfolge im Chart

        return {"ticker": ticker_symbol, "sentiments": sentiments}
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Sentiment-Daten für {ticker_symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden der Sentiment-Daten für {ticker_symbol}")

@app.get("/api/v1/potential_buys")
async def get_potential_buys(db: Session = Depends(get_db)):
    potential_buys_list = []
    try:
        # Lese die aktuelle Einstellung für Handelsaggressivität
        aggressiveness_setting = db.execute(
            text("SELECT setting_value FROM bot_settings WHERE setting_key = 'trade_aggressiveness'")
        ).fetchone()
        
        trade_aggressiveness = 5 # Standardwert, falls nicht in DB gefunden oder ungültig
        if aggressiveness_setting and aggressiveness_setting.setting_value:
            try:
                trade_aggressiveness = int(aggressiveness_setting.setting_value)
            except ValueError:
                logging.warning(f"Ungültiger Wert für trade_aggressiveness in bot_settings: '{aggressiveness_setting.setting_value}'. Verwende Standardwert 5.")
        
        # Definiere den Kaufschwellenwert basierend auf der Aggressivität
        # Je höher die Aggressivität, desto geringer der benötigte prozentuale Anstieg
        buy_threshold_multiplier = 1.02 # Standard für Aggressivität 5-6 (2% Anstieg)
        if 1 <= trade_aggressiveness <= 2: # Sehr konservativ
            buy_threshold_multiplier = 1.05 # Benötigt 5% Anstieg
        elif 3 <= trade_aggressiveness <= 4:
            buy_threshold_multiplier = 1.035 # Benötigt 3.5% Anstieg
        elif 7 <= trade_aggressiveness <= 8:
            buy_threshold_multiplier = 1.01 # Benötigt 1% Anstieg
        elif 9 <= trade_aggressiveness <= 10: # Sehr aggressiv
            buy_threshold_multiplier = 1.005 # Benötigt 0.5% Anstieg
        logging.info(f"Potenzielle Käufe: Verwende Aggressivität {trade_aggressiveness} mit Schwellenwert-Multiplikator {buy_threshold_multiplier}")

        # Vereinfacht: Nächster Handelstag ist morgen. Für eine robustere Lösung müsste man Wochenenden/Feiertage prüfen.
        # Wir suchen Vorhersagen für die nächsten 1-2 Tage, um flexibler zu sein.
        start_date = datetime.now().date() + timedelta(days=1)
        end_date = start_date + timedelta(days=2) # Vorhersagen für die nächsten 1-2 Tage

        predictions_query = text("""
            SELECT mp.ticker, mp.predicted_value, mp.prediction_date,
                   (SELECT sdc.close FROM stock_data_ohlc sdc WHERE sdc.ticker = mp.ticker ORDER BY sdc.date DESC LIMIT 1) as current_price
            FROM model_predictions mp
            WHERE mp.prediction_date BETWEEN :start_date AND :end_date
              AND mp.predicted_value IS NOT NULL
            ORDER BY mp.ticker, mp.prediction_date;
        """)
        predictions_result = db.execute(predictions_query, {"start_date": start_date, "end_date": end_date}).fetchall()

        positions_result = db.execute(text("SELECT symbol FROM positions;")).fetchall()
        owned_tickers = {row.symbol for row in positions_result}

        processed_tickers_for_potential = set() # Um Duplikate pro Ticker zu vermeiden, wenn mehrere Vorhersagen existieren

        for pred_row_proxy in predictions_result:
            pred_row = dict(pred_row_proxy._mapping) # Konvertiere zu Dict
            ticker = pred_row['ticker']
            if ticker in owned_tickers or ticker in processed_tickers_for_potential:
                continue

            current_price_val = pred_row.get('current_price')
            predicted_price_val = pred_row.get('predicted_value')

            if current_price_val is None:
                logging.warning(f"Potenzielle Käufe: Konnte aktuellen Preis für {ticker} nicht aus stock_data_ohlc ermitteln. Überspringe Ticker {ticker}.")
                continue
            
            if predicted_price_val is None: # Sollte durch Query bereits ausgeschlossen sein (IS NOT NULL)
                logging.warning(f"Potenzielle Käufe: Konnte vorhergesagten Preis für {ticker} nicht ermitteln. Überspringe Ticker {ticker}.")
                continue

            try:
                current_price = float(current_price_val)
                predicted_price = float(predicted_price_val)
            except ValueError as e:
                logging.error(f"Potenzielle Käufe: Konnte Preise für {ticker} nicht in float konvertieren. current_price='{current_price_val}', predicted_price='{predicted_price_val}'. Fehler: {e}. Überspringe Ticker {ticker}.")
                continue

            # Einfache Logik: Kaufe, wenn Vorhersage z.B. > 2% über aktuellem Preis
            if current_price and predicted_price > current_price * buy_threshold_multiplier:
                buy_info = {
                    "ticker": ticker,
                    "current_price": current_price, # Bereits float
                    "predicted_price": predicted_price, # Bereits float
                    "prediction_date": str(pred_row['prediction_date']) # Datum explizit als String
                    # Weitere Felder aus pred_row hier hinzufügen, falls benötigt
                }
                potential_buys_list.append(buy_info)
                processed_tickers_for_potential.add(ticker)
    except Exception as e:
        logging.error(f"Fehler beim Ermitteln potenzieller Käufe: {e}", exc_info=True)
        # Nicht unbedingt ein HTTP-Error hier, da die Hauptseite trotzdem laden soll
    return {"potential_buys": potential_buys_list}

class BotSettingUpdate(BaseModel):
    setting_key: str
    setting_value: str

@app.post("/api/v1/bot_settings")
async def update_bot_setting(setting_update: BotSettingUpdate, db: Session = Depends(get_db)):
    try:
        stmt = text("""
            INSERT INTO bot_settings (setting_key, setting_value)
            VALUES (:key, :value)
            ON CONFLICT (setting_key) DO UPDATE SET
                setting_value = EXCLUDED.setting_value,
                updated_at = CURRENT_TIMESTAMP;
        """)
        db.execute(stmt, {"key": setting_update.setting_key, "value": setting_update.setting_value})
        db.commit()
        return {"message": f"Einstellung {setting_update.setting_key} aktualisiert."}
    except Exception as e:
        db.rollback()
        logging.error(f"Fehler beim Aktualisieren der Einstellung {setting_update.setting_key}: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Speichern der Einstellung.")

@app.post("/api/v1/trigger_retraining/{ticker_symbol}")
async def trigger_manual_retraining(ticker_symbol: str, db: Session = Depends(get_db)):
    try:
        # Prüfen, ob der Ticker in target_tickers ist
        target_ticker_check = db.execute(
            text("SELECT 1 FROM target_tickers WHERE ticker = :ticker"),
            {"ticker": ticker_symbol}
        ).fetchone()
        if not target_ticker_check:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker_symbol} nicht in target_tickers gefunden. Training nicht möglich.")

        # Flag setzen oder aktualisieren
        stmt = text("""
            INSERT INTO retraining_flags (ticker, reason, status, flagged_at, last_checked_by_trainer)
            VALUES (:ticker, :reason, :status, CURRENT_TIMESTAMP, NULL)
            ON CONFLICT (ticker) DO UPDATE SET
                reason = EXCLUDED.reason,
                status = EXCLUDED.status,
                flagged_at = CURRENT_TIMESTAMP,
                last_checked_by_trainer = NULL; 
        """)
        db.execute(stmt, {"ticker": ticker_symbol, "reason": "manual_trigger_via_api", "status": "pending"})
        db.commit()
        logging.info(f"Manuelles Retraining für {ticker_symbol} via API ausgelöst und zur Warteschlange hinzugefügt.")
        return {"message": f"Retraining für {ticker_symbol} wurde zur Warteschlange hinzugefügt."}
    except HTTPException:
        raise # HTTPException direkt weiterleiten
    except Exception as e:
        db.rollback()
        logging.error(f"Fehler beim Auslösen des manuellen Retrainings für {ticker_symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Interner Serverfehler beim Auslösen des Trainings für {ticker_symbol}.")

@app.get("/api/time")
async def get_current_time():
    # Gibt die aktuelle Serverzeit im gewünschten Format zurück
    now = datetime.now()
    return {"time": now.strftime("%Y-%m-%d %H:%M:%S")}