import sqlite3
from datetime import datetime

DB_PATH = './sentihealth.db'

def init_notifications_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product     TEXT NOT NULL,
            type        TEXT NOT NULL,
            message     TEXT NOT NULL,
            is_read     INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_notification(product, notif_type, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO notifications (product, type, message, is_read, created_at)
        VALUES (?, ?, ?, 0, ?)
    ''', (product, notif_type, message, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_notifications(unread_only=True):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if unread_only:
        c.execute('''
            SELECT id, product, type, message, is_read, created_at
            FROM notifications
            WHERE is_read = 0
            ORDER BY created_at DESC
        ''')
    else:
        c.execute('''
            SELECT id, product, type, message, is_read, created_at
            FROM notifications
            ORDER BY created_at DESC
            LIMIT 50
        ''')
    rows = c.fetchall()
    conn.close()
    return rows

def mark_as_read(notification_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if notification_id:
        c.execute('UPDATE notifications SET is_read = 1 WHERE id = ?', (notification_id,))
    else:
        c.execute('UPDATE notifications SET is_read = 1')
    conn.commit()
    conn.close()

def get_recent_negative_count(product, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT sentiment FROM predictions
        WHERE product = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (product, limit))
    rows = c.fetchall()
    conn.close()
    return sum(1 for r in rows if r[0] == 'negative')

def get_positive_percentage(product):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT sentiment, COUNT(*) FROM predictions
        WHERE product = ?
        GROUP BY sentiment
    ''', (product,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return None
    total = sum(r[1] for r in rows)
    positive = sum(r[1] for r in rows if r[0] == 'positive')
    return round((positive / total) * 100, 1)

def get_total_reviews(product):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM predictions WHERE product = ?', (product,))
    count = c.fetchone()[0]
    conn.close()
    return count