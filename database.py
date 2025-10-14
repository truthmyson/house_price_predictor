import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def init_db():
    """Initialize the SQLite database"""
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                area FLOAT NOT NULL,
                bedrooms INTEGER NOT NULL,
                bathrooms REAL NOT NULL,
                stories INTEGER NOT NULL,
                mainroad INTEGER NOT NULL,
                guestroom INTEGER NOT NULL,
                basement INTEGER NOT NULL,
                hotwaterheating INTEGER NOT NULL,
                airconditioning INTEGER NOT NULL,
                parking INTEGER NOT NULL,
                prefarea INTEGER NOT NULL,
                furnishingstatus TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")


def save_prediction_data(features, predicted_price):
    """Save prediction data to the database"""
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                name, area, bedrooms, bathrooms, stories, mainroad, guestroom,
                basement, hotwaterheating, airconditioning, parking, prefarea,
                furnishingstatus, predicted_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            features['name'],
            features['area'],
            features['bedrooms'],
            features['bathrooms'],
            features['stories'],
            features['mainroad'],
            features['guestroom'],
            features['basement'],
            features['hotwaterheating'],
            features['airconditioning'],
            features['parking'],
            features['prefarea'],
            features['furnishingstatus'],
            predicted_price
        ))
        
        conn.commit()
        conn.close()
        logger.info("Prediction data saved to database")
    except Exception as e:
        logger.error(f"Error saving to database: {str(e)}")
