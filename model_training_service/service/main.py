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

def get_db_connection():
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
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
        # Wir holen die letzten 5 Jahre, um genügend Daten für das Training zu haben
        # Passe 'date' > NOW() - INTERVAL '5 years' an, falls nötig
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
        # Wir holen alle verfügbaren Sentiment-Daten für den Ticker
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
            # Umbenennen für einfacheres Mergen, falls nötig
            sentiment_df.rename(columns={'data_date': 'date'}, inplace=True)
            sentiment_df.set_index('date', inplace=True)
            logging.info(f"{len(sentiment_df)} Sentiment-Datensätze für {ticker_symbol} geladen.")

    except Exception as e:
        logging.error(f"Fehler beim Laden der Daten für {ticker_symbol}: {e}")
        return None, None # Gebe None zurück, wenn ein Fehler auftritt

    return ohlc_df, sentiment_df

def preprocess_data(ohlc_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Bereitet die Daten für das Modelltraining vor.
    - Merged OHLC und Sentiment Daten.
    - Behandelt fehlende Werte.
    - Erstellt Features (z.B. Returns, technische Indikatoren - später).
    - Normalisiert die Daten (später).
    """
    if ohlc_df is None or ohlc_df.empty:
        logging.warning("OHLC-Daten sind leer oder None, Vorverarbeitung nicht möglich.")
        return None

    # Kopiere das DataFrame, um Originaldaten nicht zu verändern
    data = ohlc_df.copy()

    # Wähle relevante Spalten für das Training aus
    # Wir nehmen 'open', 'high', 'low', 'close', 'volume' und 'sentiment_score'
    feature_columns = ['open', 'high', 'low', 'close', 'volume']

    # Merge mit Sentiment-Daten, falls vorhanden
    if sentiment_df is not None and not sentiment_df.empty:
        # Für den Anfang mergen wir auf das Datum.
        # Wenn mehrere Sentiment-Quellen pro Tag existieren, müssen wir entscheiden, wie wir damit umgehen (z.B. Durchschnitt)
        # Hier nehmen wir an, dass sentiment_df bereits aggregiert ist oder wir den ersten/letzten Eintrag nehmen.
        # Für den Moment: einfacher Left-Merge
        data = data.merge(sentiment_df[['sentiment_score']], left_index=True, right_index=True, how='left')
        # Fülle fehlende Sentiment-Werte (z.B. für Tage ohne Sentiment-Update oder Wochenenden)
        # Eine Möglichkeit ist Forward-Fill, um das letzte bekannte Sentiment zu verwenden.
        # Eine andere ist, sie als neutral (0) zu behandeln oder einen speziellen Wert.
        data['sentiment_score'] = data['sentiment_score'].fillna(0) # Fülle NaNs mit 0 (neutral)
        logging.info("Sentiment-Daten gemerged und NaNs mit 0 gefüllt.")
        if 'sentiment_score' not in feature_columns:
            feature_columns.append('sentiment_score')
    else:
        # Wenn keine Sentiment-Daten vorhanden sind, erstelle eine Spalte mit Nullen
        data['sentiment_score'] = 0
        logging.info("Keine Sentiment-Daten vorhanden, 'sentiment_score' mit 0 initialisiert.")
        if 'sentiment_score' not in feature_columns: # Sollte schon drin sein, aber zur Sicherheit
             feature_columns.append('sentiment_score')


    # Stelle sicher, dass die wichtigsten Spalten numerisch sind und keine NaNs enthalten (außer Sentiment)
    # adj_close wird oft für Berechnungen verwendet, aber für die Vorhersage nehmen wir die direkten Preise.
    # 'close' wird unsere primäre Zielvariable sein.
    cols_to_check = ['open', 'high', 'low', 'close', 'volume'] # adj_close ist nicht mehr in feature_columns
    for col in cols_to_check:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=cols_to_check, inplace=True) # Entferne Zeilen, wo Kern-OHLCV-Daten fehlen

    # Behalte nur die ausgewählten Feature-Spalten und die Zielspalte ('close')
    # Die Zielvariable ist der 'close'-Preis, den wir vorhersagen wollen.
    # Die Features sind die Spalten in `feature_columns`.
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
    """
    Erstellt Sequenzen von Datenpunkten und die zugehörigen Zielwerte.
    :param data: DataFrame mit den Features. Der 'close'-Preis muss die Spalte sein, die vorhergesagt wird.
                 Es wird angenommen, dass die 'close'-Spalte an Index 3 ist (open, high, low, close, ...).
    :param sequence_length: Länge der Eingabesequenz (Anzahl der vergangenen Tage).
    :param forecast_horizon: Anzahl der Tage in die Zukunft für die Vorhersage (z.B. 7 für 7 Tage).
    :return: Tuple aus (Sequenzen, Ziele).
    """
    xs, ys = [], []
    # Stelle sicher, dass 'close' in den Spalten ist, um den Index zu finden
    if 'close' not in data.columns:
        raise ValueError("Die Spalte 'close' muss im DataFrame für die Zielerstellung vorhanden sein.")
    
    # Wir verwenden die Werte des DataFrames direkt (numpy array)
    # Die Zielvariable ist der 'close'-Preis.
    # Wir nehmen an, dass die Spaltenreihenfolge in `data` der in `feature_columns` entspricht.
    # Und dass 'close' eine dieser Spalten ist.
    
    # Wir brauchen die 'close'-Spalte für das Ziel y
    # Die Eingabe X sind alle Features der Sequenzlänge
    # Das Ziel y ist der 'close'-Preis `forecast_horizon` Tage nach dem Ende der Sequenz
    
    # Konvertiere das DataFrame in ein NumPy-Array für schnellere Verarbeitung
    data_np = data.values
    close_price_index = data.columns.get_loc('close')

    for i in range(len(data_np) - sequence_length - forecast_horizon + 1):
        x = data_np[i:(i + sequence_length)]
        y = data_np[i + sequence_length + forecast_horizon - 1, close_price_index] # Close-Preis am Vorhersagetag
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def split_and_scale_data(X: np.ndarray, y: np.ndarray, train_split_ratio: float = 0.8) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler, MinMaxScaler
]:
    """
    Teilt Daten in Trainings- und Testsets auf und skaliert sie.
    """
    # Aufteilung in Trainings- und Testdaten
    train_size = int(len(X) * train_split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Skalierung der Features (X)
    # Wichtig: Scaler auf Trainingsdaten anpassen und dann auf Testdaten anwenden
    # X_train hat die Form (num_samples, sequence_length, num_features)
    # Wir müssen es für den Scaler umformen und dann zurückformen.
    num_samples_train, seq_len_train, num_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features_train)
    
    scaler_x = MinMaxScaler()
    X_train_scaled_reshaped = scaler_x.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(num_samples_train, seq_len_train, num_features_train)

    num_samples_test, seq_len_test, num_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_features_test)
    X_test_scaled_reshaped = scaler_x.transform(X_test_reshaped) # Wichtig: transform, nicht fit_transform
    X_test_scaled = X_test_scaled_reshaped.reshape(num_samples_test, seq_len_test, num_features_test)

    # Skalierung der Zielvariable (y)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Konvertierung zu PyTorch Tensoren
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_x, scaler_y

# --- PyTorch Modell Definition ---
class TransformerLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers_lstm: int, 
                 nhead_transformer: int, num_encoder_layers_transformer: int, dim_feedforward_transformer: int, 
                 output_dim: int = 1, dropout: float = 0.1):
        super(TransformerLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # Wird für LSTM verwendet

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, # Die Anzahl der erwarteten Features in der Eingabe
            nhead=nhead_transformer,
            dim_feedforward=dim_feedforward_transformer,
            dropout=dropout,
            batch_first=True # Wichtig, da unsere Daten (batch, seq_len, features) sind
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers_transformer)

        # LSTM Layer
        # Die Eingabe für LSTM ist die Ausgabe des Transformers.
        # Die Ausgabe des TransformerEncoders hat dieselbe Dimension wie die Eingabe (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim, # Output-Dim des Transformers ist gleich input_dim
            hidden_size=hidden_dim,
            num_layers=num_layers_lstm,
            batch_first=True, # Wichtig
            dropout=dropout if num_layers_lstm > 1 else 0
        )

        # Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        
        # Transformer Encoder
        # Für den Transformer Encoder benötigen wir keine Maske, wenn wir die gesamte Sequenz auf einmal verarbeiten.
        transformer_out = self.transformer_encoder(x)
        # transformer_out shape: (batch_size, seq_len, input_dim)

        # LSTM
        # Wir verwenden die gesamte Sequenzausgabe des Transformers als Eingabe für LSTM
        lstm_out, (hidden, cell) = self.lstm(transformer_out)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Wir wollen nur die Ausgabe des letzten Zeitschritts der LSTM für die Vorhersage verwenden
        last_time_step_out = lstm_out[:, -1, :]
        # last_time_step_out shape: (batch_size, hidden_dim)

        # Fully Connected Layer
        out = self.fc(last_time_step_out)
        # out shape: (batch_size, output_dim)
        return out

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, 
                ticker: str, num_epochs: int, batch_size: int, learning_rate: float):
    logging.info(f"Starte Training für {ticker}...")
    criterion = nn.MSELoss() # Mean Squared Error für Regressionsprobleme
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle für Training

    # Gerät auswählen (GPU, falls verfügbar, sonst CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Training für {ticker} auf Gerät: {device}")

    for epoch in range(num_epochs):
        model.train() # Setze Modell in den Trainingsmodus
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0: # Logge alle 10 Epochen und die erste
            logging.info(f"Ticker: {ticker}, Epoche [{epoch+1}/{num_epochs}], Durchschnittlicher Verlust: {avg_epoch_loss:.6f}")
    logging.info(f"Training für {ticker} abgeschlossen.")

def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, 
                   scaler_y: MinMaxScaler, ticker: str, batch_size: int):
    logging.info(f"Starte Evaluierung für {ticker}...")
    model.eval() # Setze Modell in den Evaluationsmodus
    criterion = nn.MSELoss()
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Kein Shuffle für Test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Stelle sicher, dass das Modell auf dem richtigen Gerät ist

    total_loss = 0.0
    all_predictions_scaled = []
    all_targets_scaled = []

    with torch.no_grad(): # Keine Gradientenberechnung während der Evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            
            all_predictions_scaled.extend(outputs.cpu().numpy())
            all_targets_scaled.extend(target.cpu().numpy())

    avg_test_loss = total_loss / len(test_loader)
    logging.info(f"Ticker: {ticker}, Durchschnittlicher Test-Verlust (MSE, skaliert): {avg_test_loss:.6f}")

    # Optional: Inverse Transformation der Vorhersagen und Ziele, um den Fehler in der Originalskala zu sehen
    # predictions_actual_scale = scaler_y.inverse_transform(np.array(all_predictions_scaled).reshape(-1,1))
    # targets_actual_scale = scaler_y.inverse_transform(np.array(all_targets_scaled).reshape(-1,1))
    # mse_actual_scale = np.mean((predictions_actual_scale - targets_actual_scale)**2)
    # logging.info(f"Ticker: {ticker}, Durchschnittlicher Test-Verlust (MSE, Originalskala): {mse_actual_scale:.6f}")

def generate_and_store_predictions(
    conn, model: nn.Module, latest_data_df: pd.DataFrame, 
    scaler_x: MinMaxScaler, scaler_y: MinMaxScaler, 
    ticker: str, sequence_length: int, forecast_horizon: int, model_name: str
):
    logging.info(f"Generiere Vorhersage für {ticker} für die nächsten {forecast_horizon} Tage...")
    model.eval() # Evaluationsmodus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(latest_data_df) < sequence_length:
        logging.warning(f"Nicht genügend aktuelle Daten ({len(latest_data_df)} Punkte) für {ticker}, um eine Sequenz der Länge {sequence_length} zu erstellen. Überspringe Vorhersage.")
        return

    # Nimm die letzten 'sequence_length' Datenpunkte
    # Die Spalten in latest_data_df sollten mit denen übereinstimmen, die für das Training verwendet wurden
    # (open, high, low, close, volume, sentiment_score)
    input_sequence_np = latest_data_df.iloc[-sequence_length:].values
    
    # Skaliere die Eingabesequenz
    input_sequence_scaled = scaler_x.transform(input_sequence_np)
    
    # Konvertiere zu Tensor und füge Batch-Dimension hinzu
    input_tensor = torch.tensor(input_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # Inverse Transformation der Vorhersage
    prediction_actual_scale = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())[0,0]

    # Bestimme das Datum der Vorhersage
    # Das ist `forecast_horizon` Tage nach dem letzten Datum in `latest_data_df`
    last_date = latest_data_df.index[-1]
    # Wichtig: pd.Timedelta erwartet eine Einheit. pd.to_datetime kann auch mit Business Days umgehen,
    # aber für eine einfache Verschiebung ist Timedelta ausreichend.
    # Beachte, dass dies Kalendertage sind. Für Handelstage wäre eine komplexere Logik nötig.
    prediction_target_date = last_date + pd.Timedelta(days=forecast_horizon)

    logging.info(f"Ticker: {ticker}, Vorhersage für {prediction_target_date.strftime('%Y-%m-%d')}: {prediction_actual_scale:.2f}")

    # Speichere die Vorhersage in der Datenbank
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_predictions 
                    (ticker, prediction_date, predicted_value, model_name, sequence_length, forecast_horizon)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, prediction_date, model_name, sequence_length, forecast_horizon) DO UPDATE SET
                    predicted_value = EXCLUDED.predicted_value,
                    generated_at = CURRENT_TIMESTAMP;
            """, (ticker, prediction_target_date.date(), float(prediction_actual_scale), model_name, sequence_length, forecast_horizon))
            conn.commit()
        logging.info(f"Vorhersage für {ticker} am {prediction_target_date.strftime('%Y-%m-%d')} in DB gespeichert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Vorhersage für {ticker}: {e}")
        if conn:
            conn.rollback()

def get_pending_retraining_flags(conn) -> list:
    """Holt Ticker, die für ein Retraining mit Status 'pending' markiert sind."""
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
    """Aktualisiert den Status eines Retraining-Flags."""
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
    # Fürs Fine-Tuning könnten wir eine geringere Lernrate oder weniger Epochen verwenden.
    # Hier verwenden wir dieselbe Trainingsfunktion, aber man könnte sie anpassen.
    logging.info(f"Starte Fine-Tuning für {ticker}...")
    # Die `train_model` Funktion kann hier wiederverwendet werden.
    # Ggf. mit angepassten Parametern für Fine-Tuning (z.B. weniger Epochen, kleinere Lernrate)
    train_model(model, X_train, y_train, ticker, num_epochs=num_epochs // 2, batch_size=batch_size, learning_rate=learning_rate / 5)
    logging.info(f"Fine-Tuning für {ticker} abgeschlossen.")

def main_training_logic():
    logging.info("Model Training Service startet...")
    conn = None
    try:
        conn = get_db_connection()
        create_db_tables_if_not_exist(conn) # Stelle sicher, dass die Tabelle existiert

        processed_tickers_in_this_run = set()

        # Hole Ticker aus der target_tickers Tabelle
        tickers_to_train = []
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT ticker FROM target_tickers ORDER BY ticker ASC;")
            rows = cur.fetchall()
            tickers_to_train = [row['ticker'] for row in rows]

        if not tickers_to_train:
            logging.info("Keine Ticker zum Trainieren in 'target_tickers' gefunden.")
            return

        # 1. Bearbeite Ticker, die für Retraining/Fine-Tuning markiert sind
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
            forecast_horizon = 7
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

            # Für Fine-Tuning verwenden wir alle verfügbaren neuen Daten zum Training (kein Test-Split hier, oder ein sehr kleiner)
            # Oder wir könnten den Test-Split beibehalten, um die Verbesserung zu messen.
            # Hier: Einfachheitshalber verwenden wir alle neuen Daten zum Fine-Tuning.
            # Skalierung muss mit den *Original-Scalern* erfolgen, wenn möglich, oder neu angepasst werden.
            # Das Laden und Wiederverwenden von Scalern ist komplexer.
            # Für den Moment: Wir skalieren die neuen Daten einfach.
            # ACHTUNG: Dies ist eine Vereinfachung. Idealerweise sollten Scaler gespeichert und geladen werden.
            
            # Da wir die Scaler nicht speichern, müssen wir sie hier neu erstellen.
            # Das bedeutet, dass die Skalierung möglicherweise nicht 100% konsistent mit dem ursprünglichen Training ist.
            # Eine bessere Lösung wäre, die Scaler zusammen mit dem Modell zu speichern.
            num_samples, seq_len, num_features = X.shape
            X_reshaped = X.reshape(-1, num_features)
            temp_scaler_x = MinMaxScaler()
            X_scaled_reshaped = temp_scaler_x.fit_transform(X_reshaped)
            X_scaled = X_scaled_reshaped.reshape(num_samples, seq_len, num_features)
            temp_scaler_y = MinMaxScaler()
            y_scaled = temp_scaler_y.fit_transform(y.reshape(-1, 1))

            X_fine_tune_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_fine_tune_tensor = torch.tensor(y_scaled, dtype=torch.float32)

            # Modell laden
            current_model_name = "TransformerLSTM_v1" # Muss konsistent sein
            model_path = os.path.join("/app/trained_models", f"{ticker}_{current_model_name}.pth")
            
            if not os.path.exists(model_path):
                logging.warning(f"Kein vortrainiertes Modell für {ticker} unter {model_path} gefunden. Führe stattdessen volles Training durch.")
                update_retraining_flag_status(conn, ticker, 'pending', 'Vortrainiertes Modell nicht gefunden, volles Training nötig')
                # Ticker wird dann im normalen Trainingslauf behandelt
                continue # Nicht zu processed_tickers_in_this_run hinzufügen, damit es neu trainiert wird

            # Modellparameter (müssen mit dem gespeicherten Modell übereinstimmen)
            input_dim = X_fine_tune_tensor.shape[2]
            # ... (restliche Modellparameter wie beim ursprünglichen Training definieren)
            # Diese Parameter müssen entweder gespeichert/geladen oder hier konsistent definiert werden.
            # Annahme: Parameter sind dieselben wie unten im Haupttraining.
            hidden_dim_lstm = 128; num_layers_lstm = 2; nhead_transformer = input_dim; num_encoder_layers_transformer = 2; dim_feedforward_transformer = 256; output_dim = 1
            if input_dim % nhead_transformer != 0:
                possible_nheads = [h for h in range(1, input_dim + 1) if input_dim % h == 0]; nhead_transformer = possible_nheads[-1] if possible_nheads else 1

            model = TransformerLSTMModel(input_dim, hidden_dim_lstm, num_layers_lstm, nhead_transformer, num_encoder_layers_transformer, dim_feedforward_transformer, output_dim)
            try:
                model.load_state_dict(torch.load(model_path))
                logging.info(f"Vortrainiertes Modell für {ticker} geladen von {model_path}")

                # Fine-Tuning Parameter
                num_epochs_finetune = 25 # Weniger Epochen für Fine-Tuning
                batch_size_finetune = 32
                learning_rate_finetune = 0.0002 # Kleinere Lernrate

                fine_tune_model(model, X_fine_tune_tensor, y_fine_tune_tensor, ticker, num_epochs_finetune, batch_size_finetune, learning_rate_finetune)
                
                # Speichere das fein-getunte Modell
                torch.save(model.state_dict(), model_path) # Überschreibe das alte Modell
                logging.info(f"Fein-getuntes Modell für {ticker} gespeichert unter: {model_path}")

                # Generiere neue Vorhersagen mit dem fein-getunten Modell
                # Wichtig: scaler_x und scaler_y für generate_and_store_predictions müssen die neu erstellten temp_scaler sein.
                generate_and_store_predictions(conn, model, processed_data, temp_scaler_x, temp_scaler_y, ticker, sequence_length, forecast_horizon, current_model_name)
                update_retraining_flag_status(conn, ticker, 'completed')
            except Exception as e_ft:
                logging.error(f"Fehler beim Fine-Tuning von {ticker}: {e_ft}", exc_info=True)
                update_retraining_flag_status(conn, ticker, 'error', f"Fine-Tuning Fehler: {e_ft}")
            
            processed_tickers_in_this_run.add(ticker)
            time.sleep(2)

        # 2. Reguläres Training für alle Ticker (außer denen, die gerade fein-getunt wurden)
        logging.info(f"\n--- Starte regulären Trainingsdurchlauf ---")
        logging.info(f"Ticker aus target_tickers: {tickers_to_train}")
        logging.info(f"Bereits in diesem Lauf verarbeitet (z.B. Fine-Tuning): {processed_tickers_in_this_run}")
        
        for ticker in tickers_to_train:
            if ticker in processed_tickers_in_this_run:
                logging.info(f"Ticker {ticker} wurde bereits in diesem Durchlauf verarbeitet (z.B. Fine-Tuning). Überspringe reguläres Training.")
                continue

            logging.info(f"\n--- Starte Verarbeitung für Ticker: {ticker} ---")
            ohlc_data, sentiment_data = load_data_for_ticker(conn, ticker)

            if ohlc_data is None or ohlc_data.empty:
                logging.warning(f"Keine ausreichenden OHLC-Daten für {ticker} zum Trainieren. Überspringe.")
                continue

            processed_data = preprocess_data(ohlc_data, sentiment_data)
            if processed_data is None or processed_data.empty:
                logging.warning(f"Daten für {ticker} konnten nicht vorverarbeitet werden oder sind leer. Überspringe.")
                continue
            
            # `processed_data` ist das DataFrame mit den Features, die für Sequenzen und Vorhersagen verwendet werden.

            # Parameter für Sequenzerstellung und Vorhersage
            sequence_length = 60  # Anzahl der vergangenen Tage als Input
            forecast_horizon = 7    # Vorhersage für die nächsten 7 Tage (wir nehmen den Close-Preis am 7. Tag)

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

            # Daten aufteilen und skalieren
            logging.info(f"Teile und skaliere Daten für {ticker}...")
            X_train, y_train, X_test, y_test, scaler_x, scaler_y = split_and_scale_data(X, y)
            logging.info(f"Daten aufgeteilt und skaliert: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Modellparameter
            input_dim = X_train.shape[2]  # Anzahl der Features
            hidden_dim_lstm = 128
            num_layers_lstm = 2
            nhead_transformer = input_dim # Muss ein Teiler von input_dim sein, oder input_dim selbst wenn input_dim klein ist.
                                          # Sicherstellen, dass input_dim % nhead_transformer == 0.
                                          # Wenn input_dim z.B. 6 ist, kann nhead 1, 2, 3, 6 sein.
            if input_dim % nhead_transformer != 0: # Einfache Anpassung, falls nicht teilbar
                # Finde den größten Teiler von input_dim oder setze auf 1
                possible_nheads = [h for h in range(1, input_dim + 1) if input_dim % h == 0]
                nhead_transformer = possible_nheads[-1] if possible_nheads else 1
                logging.warning(f"nhead_transformer angepasst auf {nhead_transformer} für input_dim {input_dim}")

            num_encoder_layers_transformer = 2
            dim_feedforward_transformer = 256
            output_dim = 1 # Wir sagen einen einzelnen Wert voraus (Close-Preis)

            model = TransformerLSTMModel(input_dim, hidden_dim_lstm, num_layers_lstm,
                                         nhead_transformer, num_encoder_layers_transformer,
                                         dim_feedforward_transformer, output_dim)
            logging.info(f"Modell für {ticker} erstellt: {model}")

            # Trainingsparameter
            num_epochs = 50 # Beispiel: Anzahl der Epochen
            batch_size = 32  # Beispiel: Batch-Größe
            learning_rate = 0.001 # Beispiel: Lernrate
            current_model_name = "TransformerLSTM_v1" # Könnte versioniert werden

            train_model(model, X_train, y_train, ticker, num_epochs, batch_size, learning_rate)

            # Evaluierung des Modells
            evaluate_model(model, X_test, y_test, scaler_y, ticker, batch_size)

            # Speichern des trainierten Modells
            # Erstelle einen Ordner für Modelle, falls er nicht existiert
            models_dir = "/app/trained_models"
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{ticker}_{current_model_name}.pth")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Modell für {ticker} gespeichert unter: {model_path}")

            # Generiere und speichere Vorhersagen
            # `processed_data` enthält alle historischen Daten inklusive der Features.
            # Wir verwenden dieses DataFrame, um die letzte Sequenz für die Vorhersage zu erhalten.
            generate_and_store_predictions(
                conn, model, processed_data, scaler_x, scaler_y, 
                ticker, sequence_length, forecast_horizon, current_model_name)

            logging.info(f"--- Verarbeitung für Ticker: {ticker} abgeschlossen ---")
            # Kurze Pause, um Logs lesbar zu halten und Ressourcen nicht zu überlasten
            time.sleep(2)


    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Model Training Service: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Datenbankverbindung geschlossen.")

    logging.info("Model Training Service hat seine Aufgabe beendet.")


if __name__ == "__main__":
    # Für den Anfang lassen wir es einmalig laufen.
    # Periodische Ausführung, z.B. alle 24 Stunden
    # Das Intervall sollte so gewählt werden, dass neue Daten (OHLC, Sentiment) wahrscheinlich verfügbar sind.
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
            # exc_info=True loggt den Stacktrace
            logging.error(f"Schwerwiegender Fehler in der Hauptschleife des Model Training Service: {e}", exc_info=True)
            logging.info("Warte 1 Stunde vor dem nächsten Versuch.")
            time.sleep(60 * 60) # Warte 1 Stunde bei Fehler vor erneutem Versuch
