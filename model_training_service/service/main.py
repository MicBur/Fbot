import os
import time
import psycopg2
import psycopg2.extras
import pandas as pd
from dotenv import load_dotenv
import logging
from typing import Tuple, Optional # Hinzugefügt für ältere Python-Versionen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Konfiguration aus Umgebungsvariablen ---
# Diese Variablen sollten in deiner .env Datei oder über Docker Compose gesetzt werden
# Beispielwerte sind hier als Fallback angegeben.
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "trading_bot_db")
DB_USER = os.getenv("POSTGRES_USER", "youruser")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "yourpassword")

MODEL_SAVE_PATH = "/app/trained_models" # Pfad innerhalb des Containers, gemountet via Docker Volume
PREDICTION_HORIZON = int(os.getenv("PREDICTION_HORIZON", "7")) # Tage
MODEL_TYPE = os.getenv("MODEL_TYPE", "TransformerLSTM")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
FORCE_FULL_RETRAINING_ON_START = os.getenv("FORCE_FULL_RETRAINING_ON_START", "false").lower() == "true"

# --- Datenbank Helfer ---
def get_db_connection():
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port="5432")
        logging.info("Datenbankverbindung erfolgreich hergestellt.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Fehler beim Herstellen der Datenbankverbindung: {e}")
        raise

def create_db_tables_if_not_exist(conn):
    """Erstellt die notwendigen Tabellen, falls sie noch nicht existieren."""
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS model_predictions (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            prediction_date DATE NOT NULL, -- Das Datum, für das die Vorhersage gilt
            predicted_value NUMERIC NOT NULL,
            model_name VARCHAR(255), -- z.B. TransformerLSTMModel_v1
            sequence_length INT,
            forecast_horizon INT,
            generated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (ticker, prediction_date, model_name, sequence_length, forecast_horizon)
        );""")
        # retraining_flags Tabelle wird vom trading_decision_service erstellt, aber hier könnte man es auch tun.
        conn.commit()
    logging.info("Tabelle 'model_predictions' sichergestellt.")

def load_data_for_ticker(conn, ticker_symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Lädt historische Kursdaten (OHLCV) und Sentiment-Daten für einen gegebenen Ticker.
    """
    logging.info(f"Lade Daten für Ticker: {ticker_symbol}")
    ohlc_df = None
    sentiment_df = None

    try:
        # Lade historische Kursdaten
        query_ohlc = """
            SELECT date, open, high, low, close, adj_close, volume
            FROM stock_data_ohlc
            WHERE ticker = %s
            ORDER BY date ASC;
        """
        ohlc_df = pd.read_sql_query(query_ohlc, conn, params=(ticker_symbol,))
        if ohlc_df.empty:
            logging.warning(f"Keine OHLC-Daten für {ticker_symbol} gefunden.")
        else:
            ohlc_df['date'] = pd.to_datetime(ohlc_df['date'])
            ohlc_df.set_index('date', inplace=True)
            logging.info(f"{len(ohlc_df)} OHLC-Datensätze für {ticker_symbol} geladen.")

        # Lade Sentiment-Daten
        query_sentiment = """
            SELECT data_date, sentiment_score, sentiment_source
            FROM sentiment_data
            WHERE ticker = %s
            ORDER BY data_date ASC;
        """
        sentiment_df = pd.read_sql_query(query_sentiment, conn, params=(ticker_symbol,))
        if sentiment_df.empty:
            logging.warning(f"Keine Sentiment-Daten für {ticker_symbol} gefunden.")
        else:
            sentiment_df['data_date'] = pd.to_datetime(sentiment_df['data_date'])
            sentiment_df.rename(columns={'data_date': 'date'}, inplace=True)
            sentiment_df.set_index('date', inplace=True)
            logging.info(f"{len(sentiment_df)} Sentiment-Datensätze für {ticker_symbol} geladen.")

    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten für {ticker_symbol}: {e}")
        return None, None

    return ohlc_df, sentiment_df

def preprocess_data(ohlc_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if ohlc_df is None or ohlc_df.empty:
        logging.warning("OHLC-Daten sind leer oder None, Vorverarbeitung nicht möglich.")
        return None

    data = ohlc_df.copy()
    feature_columns = ['open', 'high', 'low', 'close', 'volume']

    if sentiment_df is not None and not sentiment_df.empty:
        data = data.merge(sentiment_df[['sentiment_score']], left_index=True, right_index=True, how='left')
        data['sentiment_score'] = data['sentiment_score'].fillna(0)
        logging.info("Sentiment-Daten gemerged und NaNs mit 0 gefüllt.")
        if 'sentiment_score' not in feature_columns:
            feature_columns.append('sentiment_score')
    else:
        data['sentiment_score'] = 0
        logging.info("Keine Sentiment-Daten vorhanden, 'sentiment_score' mit 0 initialisiert.")
        if 'sentiment_score' not in feature_columns:
             feature_columns.append('sentiment_score')

    cols_to_check = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_check:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=cols_to_check, inplace=True)

    if not all(col in data.columns for col in feature_columns):
        logging.error(f"Nicht alle Feature-Spalten {feature_columns} sind im DataFrame vorhanden. Verfügbare Spalten: {data.columns.tolist()}")
        return None

    processed_df = data[feature_columns].copy()

    if processed_df.empty:
        logging.warning("Daten sind nach der Vorverarbeitung leer.")
        return None

    logging.info(f"Datenvorverarbeitung abgeschlossen. Shape des Feature-DataFrames: {processed_df.shape}")
    return processed_df

def create_sequences(data: pd.DataFrame, sequence_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    if 'close' not in data.columns:
        raise ValueError("Die Spalte 'close' muss im DataFrame für die Zielerstellung vorhanden sein.")
    
    data_np = data.values
    close_price_index = data.columns.get_loc('close')

    for i in range(len(data_np) - sequence_length - forecast_horizon + 1):
        x = data_np[i:(i + sequence_length)]
        y = data_np[i + sequence_length + forecast_horizon - 1, close_price_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def split_and_scale_data(X: np.ndarray, y: np.ndarray, train_split_ratio: float = 0.8) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, MinMaxScaler
]:
    train_size = int(len(X) * train_split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    num_samples_train, seq_len_train, num_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features_train)
    
    scaler_x = MinMaxScaler()
    X_train_scaled_reshaped = scaler_x.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(num_samples_train, seq_len_train, num_features_train)

    num_samples_test, seq_len_test, num_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_features_test)
    X_test_scaled_reshaped = scaler_x.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled_reshaped.reshape(num_samples_test, seq_len_test, num_features_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_x, scaler_y

class TransformerLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers_lstm: int, 
                 nhead_transformer: int, num_encoder_layers_transformer: int, dim_feedforward_transformer: int, 
                 output_dim: int = 1, dropout: float = 0.1):
        super(TransformerLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead_transformer,
            dim_feedforward=dim_feedforward_transformer,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers_transformer)

        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout if num_layers_lstm > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformer_out = self.transformer_encoder(x)
        lstm_out, (hidden, cell) = self.lstm(transformer_out)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, 
                ticker: str, num_epochs: int, batch_size: int, learning_rate: float):
    logging.info(f"Starte Training für {ticker}...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Training für {ticker} auf Gerät: {device}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Ticker: {ticker}, Epoche [{epoch+1}/{num_epochs}], Durchschnittlicher Verlust: {avg_epoch_loss:.6f}")
    logging.info(f"Training für {ticker} abgeschlossen.")

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, 
                   scaler_y: MinMaxScaler, ticker: str, batch_size: int):
    logging.info(f"Starte Evaluierung für {ticker}...")
    model.eval()
    criterion = nn.MSELoss()
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
    avg_test_loss = total_loss / len(test_loader)
    logging.info(f"Ticker: {ticker}, Durchschnittlicher Test-Verlust (MSE, skaliert): {avg_test_loss:.6f}")

def generate_and_store_predictions(
    conn, model: nn.Module, latest_data_df: pd.DataFrame, 
    scaler_x: MinMaxScaler, scaler_y: MinMaxScaler, 
    ticker: str, sequence_length: int, forecast_horizon: int, model_name: str
):
    logging.info(f"Generiere Vorhersage für {ticker} für die nächsten {forecast_horizon} Tage...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(latest_data_df) < sequence_length:
        logging.warning(f"Nicht genügend aktuelle Daten ({len(latest_data_df)} Punkte) für {ticker}, um eine Sequenz der Länge {sequence_length} zu erstellen. Überspringe Vorhersage.")
        return

    input_sequence_np = latest_data_df.iloc[-sequence_length:].values
    input_sequence_scaled = scaler_x.transform(input_sequence_np)
    input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    prediction_actual_scale = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())[0,0]
    last_date = latest_data_df.index[-1]
    prediction_target_date = last_date + pd.Timedelta(days=forecast_horizon)

    logging.info(f"Ticker: {ticker}, Vorhersage für {prediction_target_date.strftime('%Y-%m-%d')}: {prediction_actual_scale:.2f}")

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_predictions 
                    (ticker, prediction_date, predicted_value, model_name, sequence_length, forecast_horizon)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (ticker, prediction_target_date.date(), float(prediction_actual_scale), model_name, sequence_length, forecast_horizon))
            conn.commit()
        logging.info(f"Vorhersage für {ticker} am {prediction_target_date.strftime('%Y-%m-%d')} in DB gespeichert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Vorhersage für {ticker}: {e}")
        if conn:
            conn.rollback()

def get_pending_retraining_flags(conn) -> list:
    flags = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT ticker, reason FROM retraining_flags
                WHERE status = 'pending'
                ORDER BY flagged_at ASC;
            """)
            flags = cur.fetchall()
        if flags:
            logging.info(f"{len(flags)} Ticker für Retraining gefunden: {[f['ticker'] for f in flags]}")
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Retraining-Flags: {e}")
    return flags

def update_retraining_flag_status(conn, ticker: str, status: str, reason: Optional[str] = None):
    try:
        with conn.cursor() as cur:
            if reason:
                cur.execute("""
                    UPDATE retraining_flags SET status = %s, reason = %s, last_checked_by_trainer = CURRENT_TIMESTAMP
                    WHERE ticker = %s;
                """, (status, reason, ticker))
            else:
                cur.execute("""
                    UPDATE retraining_flags SET status = %s, last_checked_by_trainer = CURRENT_TIMESTAMP
                    WHERE ticker = %s;
                """, (status, ticker))
            conn.commit()
        logging.info(f"Retraining-Flag-Status für {ticker} auf '{status}' aktualisiert.")
    except Exception as e:
        logging.error(f"Fehler beim Aktualisieren des Retraining-Flag-Status für {ticker}: {e}")
        if conn:
            conn.rollback()

def fine_tune_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, 
                    ticker: str, num_epochs: int, batch_size: int, learning_rate: float):
    logging.info(f"Starte Fine-Tuning für {ticker}...")
    train_model(model, X_train, y_train, ticker, num_epochs=num_epochs // 2, batch_size=batch_size, learning_rate=learning_rate / 5)
    logging.info(f"Fine-Tuning für {ticker} abgeschlossen.")

def main_training_logic():
    logging.info("Model Training Service startet...")
    conn = None
    try:
        conn = get_db_connection()
        create_db_tables_if_not_exist(conn)

        processed_tickers_in_this_run = set()
        tickers_to_train = []
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT ticker FROM target_tickers ORDER BY ticker ASC;")
            rows = cur.fetchall()
            tickers_to_train = [row['ticker'] for row in rows]

        if not tickers_to_train:
            logging.info("Keine Ticker zum Trainieren in 'target_tickers' gefunden.")
            return

        pending_flags = get_pending_retraining_flags(conn)
        for flag in pending_flags:
            ticker = flag['ticker']
            logging.info(f"\n--- Starte Fine-Tuning-Verarbeitung für markierten Ticker: {ticker} (Grund: {flag['reason']}) ---")
            update_retraining_flag_status(conn, ticker, 'in_progress')

            ohlc_data, sentiment_data = load_data_for_ticker(conn, ticker)
            if ohlc_data is None or ohlc_data.empty:
                logging.warning(f"Keine ausreichenden OHLC-Daten für Fine-Tuning von {ticker}. Überspringe.")
                update_retraining_flag_status(conn, ticker, 'error', 'Fehlende OHLC-Daten für Fine-Tuning')
                processed_tickers_in_this_run.add(ticker)
                continue
            
            processed_data = preprocess_data(ohlc_data, sentiment_data)
            if processed_data is None or processed_data.empty:
                logging.warning(f"Daten für Fine-Tuning von {ticker} konnten nicht vorverarbeitet werden. Überspringe.")
                update_retraining_flag_status(conn, ticker, 'error', 'Datenvorverarbeitungsfehler für Fine-Tuning')
                processed_tickers_in_this_run.add(ticker)
                continue

            sequence_length = 60
            forecast_horizon = PREDICTION_HORIZON # Verwende globale Konfiguration
            try:
                X, y = create_sequences(processed_data, sequence_length, forecast_horizon)
            except ValueError as ve:
                logging.error(f"Fehler bei Sequenzerstellung für Fine-Tuning von {ticker}: {ve}. Überspringe.")
                update_retraining_flag_status(conn, ticker, 'error', f"Sequenzerstellungsfehler: {ve}")
                processed_tickers_in_this_run.add(ticker)
                continue
            
            if X.shape[0] == 0:
                logging.warning(f"Nicht genügend Daten für Fine-Tuning von {ticker} um Sequenzen zu erstellen. Überspringe.")
                update_retraining_flag_status(conn, ticker, 'error', 'Nicht genügend Daten für Sequenzen (Fine-Tuning)')
                processed_tickers_in_this_run.add(ticker)
                continue
            
            num_samples, seq_len, num_features = X.shape
            X_reshaped = X.reshape(-1, num_features)
            temp_scaler_x = MinMaxScaler()
            X_scaled_reshaped = temp_scaler_x.fit_transform(X_reshaped)
            X_scaled = X_scaled_reshaped.reshape(num_samples, seq_len, num_features)
            temp_scaler_y = MinMaxScaler()
            y_scaled = temp_scaler_y.fit_transform(y.reshape(-1, 1))

            X_fine_tune_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_fine_tune_tensor = torch.tensor(y_scaled, dtype=torch.float32)

            current_model_name = f"{MODEL_TYPE}_v{MODEL_VERSION}"
            model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}_{current_model_name}.pth")
            
            if not os.path.exists(model_path):
                logging.warning(f"Kein vortrainiertes Modell für {ticker} unter {model_path} gefunden. Führe stattdessen volles Training durch.")
                update_retraining_flag_status(conn, ticker, 'pending', 'Vortrainiertes Modell nicht gefunden, volles Training nötig')
                continue

            input_dim = X_fine_tune_tensor.shape[2]
            hidden_dim_lstm = 128; num_layers_lstm = 2; nhead_transformer = input_dim; num_encoder_layers_transformer = 2; dim_feedforward_transformer = 256; output_dim = 1
            if input_dim > 0 and input_dim % nhead_transformer != 0: # input_dim muss > 0 sein
                possible_nheads = [h for h in range(1, input_dim + 1) if input_dim % h == 0]; nhead_transformer = possible_nheads[-1] if possible_nheads else 1

            model = TransformerLSTMModel(input_dim, hidden_dim_lstm, num_layers_lstm, nhead_transformer, num_encoder_layers_transformer, dim_feedforward_transformer, output_dim)
            try:
                model.load_state_dict(torch.load(model_path))
                logging.info(f"Vortrainiertes Modell für {ticker} geladen von {model_path}")

                num_epochs_finetune = 25
                batch_size_finetune = 32
                learning_rate_finetune = 0.0002

                fine_tune_model(model, X_fine_tune_tensor, y_fine_tune_tensor, ticker, num_epochs_finetune, batch_size_finetune, learning_rate_finetune)
                torch.save(model.state_dict(), model_path)
                logging.info(f"Fein-getuntes Modell für {ticker} gespeichert unter: {model_path}")

                generate_and_store_predictions(conn, model, processed_data, temp_scaler_x, temp_scaler_y, ticker, sequence_length, forecast_horizon, current_model_name)
                update_retraining_flag_status(conn, ticker, 'completed')
            except Exception as e_ft:
                logging.error(f"Fehler beim Fine-Tuning von {ticker}: {e_ft}", exc_info=True)
                update_retraining_flag_status(conn, ticker, 'error', f"Fine-Tuning Fehler: {e_ft}")
            
            processed_tickers_in_this_run.add(ticker)
            time.sleep(2)

        logging.info(f"\n--- Starte regulären Trainingsdurchlauf ---")
        logging.info(f"Ticker aus target_tickers: {tickers_to_train}")
        logging.info(f"Bereits in diesem Lauf verarbeitet (z.B. Fine-Tuning): {processed_tickers_in_this_run}")
        
        for ticker in tickers_to_train:
            if ticker in processed_tickers_in_this_run:
                logging.info(f"Ticker {ticker} wurde bereits in diesem Durchlauf verarbeitet. Überspringe reguläres Training.")
                continue

            # NEUE LOGIK: Modell-Existenzprüfung
            model_filename = f"{ticker}_{MODEL_TYPE}_v{MODEL_VERSION}.pth"
            model_path = os.path.join(MODEL_SAVE_PATH, model_filename)

            needs_retraining_due_to_flag = False # Wird oben schon für Fine-Tuning geprüft, hier ggf. redundant, aber schadet nicht
            if conn:
                try:
                    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("SELECT status FROM retraining_flags WHERE ticker = %s AND status = 'pending'", (ticker,))
                        flag = cur.fetchone()
                        if flag:
                            needs_retraining_due_to_flag = True
                            # Kein Log hier, da es schon beim Fine-Tuning geloggt würde, falls es daher kommt
                except Exception as e_flag:
                    logging.error(f"Fehler beim Prüfen des Retraining-Flags für {ticker} (regulärer Lauf): {e_flag}")

            if os.path.exists(model_path) and not needs_retraining_due_to_flag and not FORCE_FULL_RETRAINING_ON_START:
                logging.info(f"Modell {model_path} existiert bereits und kein Retraining-Flag oder Force-Flag aktiv. Überspringe Training für {ticker}.")
                # Optional: Hier könnte man prüfen, ob aktuelle Vorhersagen existieren und diese ggf. generieren,
                # auch wenn das Modell nicht neu trainiert wird.
                # Für den Moment: Wenn das Modell da ist und nicht neu trainiert werden muss, generieren wir auch keine neuen Vorhersagen
                # im regulären Lauf, da dies beim Fine-Tuning oder beim nächsten erzwungenen Training passiert.
                # Alternativ: Immer Vorhersagen generieren, wenn das Modell existiert.
                # Für jetzt: Überspringen, wenn Modell da und kein Retraining.
                continue
            # ENDE NEUE LOGIK

            logging.info(f"\n--- Starte Verarbeitung für Ticker: {ticker} ---")
            ohlc_data, sentiment_data = load_data_for_ticker(conn, ticker)

            if ohlc_data is None or ohlc_data.empty:
                logging.warning(f"Keine ausreichenden OHLC-Daten für {ticker} zum Trainieren. Überspringe.")
                continue

            processed_data = preprocess_data(ohlc_data, sentiment_data)
            if processed_data is None or processed_data.empty:
                logging.warning(f"Daten für {ticker} konnten nicht vorverarbeitet werden oder sind leer. Überspringe.")
                continue
            
            sequence_length = 60
            forecast_horizon = PREDICTION_HORIZON

            logging.info(f"Erstelle Sequenzen für {ticker} mit Länge {sequence_length} und Horizont {forecast_horizon}...")
            try:
                X, y = create_sequences(processed_data, sequence_length, forecast_horizon)
            except ValueError as ve:
                logging.error(f"Fehler bei Sequenzerstellung für {ticker}: {ve}. Überspringe.")
                continue

            if X.shape[0] == 0:
                logging.warning(f"Nicht genügend Daten für {ticker} um Sequenzen zu erstellen (benötigt mind. {sequence_length + forecast_horizon -1} Tage). Überspringe.")
                continue
            logging.info(f"Sequenzen erstellt: X shape: {X.shape}, y shape: {y.shape}")

            logging.info(f"Teile und skaliere Daten für {ticker}...")
            X_train, y_train, X_test, y_test, scaler_x, scaler_y = split_and_scale_data(X, y)
            logging.info(f"Daten aufgeteilt und skaliert: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            input_dim = X_train.shape[2]
            hidden_dim_lstm = 128
            num_layers_lstm = 2
            nhead_transformer = input_dim 
            if input_dim > 0 and input_dim % nhead_transformer != 0:
                possible_nheads = [h for h in range(1, input_dim + 1) if input_dim % h == 0]
                nhead_transformer = possible_nheads[-1] if possible_nheads else 1
                logging.warning(f"nhead_transformer angepasst auf {nhead_transformer} für input_dim {input_dim}")
            elif input_dim == 0: # Fallback, falls input_dim 0 ist (sollte nicht passieren bei erfolgreicher Datenverarb.)
                logging.error(f"Input_dim für {ticker} ist 0. Kann Modell nicht erstellen.")
                continue


            num_encoder_layers_transformer = 2
            dim_feedforward_transformer = 256
            output_dim = 1

            model = TransformerLSTMModel(input_dim, hidden_dim_lstm, num_layers_lstm,
                                         nhead_transformer, num_encoder_layers_transformer,
                                         dim_feedforward_transformer, output_dim)
            logging.info(f"Modell für {ticker} erstellt: {model}")

            num_epochs = 50
            batch_size = 32
            learning_rate = 0.001
            current_model_name = f"{MODEL_TYPE}_v{MODEL_VERSION}"

            train_model(model, X_train, y_train, ticker, num_epochs, batch_size, learning_rate)
            evaluate_model(model, X_test, y_test, scaler_y, ticker, batch_size)

            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            final_model_path = os.path.join(MODEL_SAVE_PATH, f"{ticker}_{current_model_name}.pth") # Verwende globale Konstante
            torch.save(model.state_dict(), final_model_path)
            logging.info(f"Modell für {ticker} gespeichert unter: {final_model_path}")

            generate_and_store_predictions(
                conn, model, processed_data, scaler_x, scaler_y, 
                ticker, sequence_length, forecast_horizon, current_model_name)

            logging.info(f"--- Verarbeitung für Ticker: {ticker} abgeschlossen ---")
            time.sleep(2)

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Model Training Service: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")
    logging.info("Model Training Service hat seine Aufgabe beendet.")

if __name__ == "__main__":
    RUN_INTERVAL_SECONDS = int(os.getenv("MODEL_TRAINING_INTERVAL_SECONDS", 24 * 60 * 60)) 
    
    while True:
        try:
            main_training_logic()
            logging.info(f"Model Training Service: Nächster Durchlauf in {RUN_INTERVAL_SECONDS / (60*60):.0f} Stunden.")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Model Training Service durch Benutzer beendet.")
            break
        except Exception as e:
            logging.error(f"Schwerwiegender Fehler in der Hauptschleife des Model Training Service: {e}", exc_info=True)
            logging.info("Warte 1 Stunde vor dem nächsten Versuch.")
            time.sleep(60 * 60)
