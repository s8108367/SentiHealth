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

def get_previous_positive_percentage(product, exclude_last=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT sentiment FROM predictions
        WHERE product = ?
        ORDER BY created_at DESC
        LIMIT 100
    ''', (product,))
    rows = c.fetchall()
    conn.close()

    if len(rows) <= exclude_last:
        return None

    older = rows[exclude_last:]
    total = len(older)
    positive = sum(1 for r in older if r[0] == 'positive')
    return round((positive / total) * 100, 1)