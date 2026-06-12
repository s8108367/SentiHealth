import sqlite3
from datetime import datetime

DB_PATH = './sentihealth.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product     TEXT,
            review      TEXT NOT NULL,
            sentiment   TEXT NOT NULL,
            confidence  REAL NOT NULL,
            created_at  TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(product, review, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (product, review, sentiment, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (product, review, sentiment, confidence, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_product_summary(product):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence
        FROM predictions
        WHERE product = ?
        GROUP BY sentiment
    ''', (product,))
    rows = c.fetchall()
    conn.close()
    return rows