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
    try:
        account_info_result = db.execute(text("SELECT * FROM account_info ORDER BY last_updated DESC LIMIT 1;")).fetchone()
        if account_info_result:
            # Konvertiere RowProxy zu einem Dict-ähnlichen Objekt, falls nötig, oder greife direkt auf Spalten zu
            account_info = dict(account_info_result._mapping)

        positions_result = db.execute(text("SELECT * FROM positions ORDER BY symbol;")).fetchall()
        positions = [dict(row._mapping) for row in positions_result]

    except Exception as e:
        error_message = f"Fehler beim Abrufen der Portfoliodaten: {e}"
        logging.error(error_message)
    
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "account_info": account_info,
        "positions": positions,
        "error_message": error_message
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

@app.get("/api/v1/chart_data/{ticker_symbol}")
async def get_chart_data(ticker_symbol: str, db: Session = Depends(get_db)):
    try:
        historical_data_query = text("""
            SELECT date, close FROM stock_data_ohlc
            WHERE ticker = :ticker ORDER BY date DESC LIMIT 365;
        """)
        historical_result = db.execute(historical_data_query, {"ticker": ticker_symbol}).fetchall()
        historical_prices = [{"date": str(row.date), "price": float(row.close)} for row in historical_result]
        historical_prices.reverse()

        prediction_data_query = text("""
            SELECT prediction_date, predicted_value FROM model_predictions
            WHERE ticker = :ticker AND prediction_date >= CURRENT_DATE ORDER BY prediction_date ASC;
        """)
        prediction_result = db.execute(prediction_data_query, {"ticker": ticker_symbol}).fetchall()
        predictions = [{"date": str(row.prediction_date), "price": float(row.predicted_value)} for row in prediction_result]

        return {"ticker": ticker_symbol, "historical": historical_prices, "predictions": predictions}
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