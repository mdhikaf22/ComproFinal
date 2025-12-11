"""
============================================================
DATABASE - SQLite Database Operations
============================================================
"""

import sqlite3
from datetime import datetime
from .config import DATABASE_PATH


def get_connection():
    """Get database connection"""
    return sqlite3.connect(DATABASE_PATH)


def init_database():
    """Initialize SQLite database for access logs"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            name VARCHAR(100),
            role VARCHAR(50),
            authorized BOOLEAN,
            confidence FLOAT,
            action VARCHAR(50),
            image_path VARCHAR(255)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_access INTEGER DEFAULT 0,
            authorized_count INTEGER DEFAULT 0,
            unauthorized_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")


def log_access(name, role, authorized, confidence, action="detection", image_path=None):
    """Log access to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO access_logs (name, role, authorized, confidence, action, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, role, authorized, confidence, action, image_path))
    
    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        INSERT INTO daily_stats (date, total_access, authorized_count, unauthorized_count)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            total_access = total_access + 1,
            authorized_count = authorized_count + ?,
            unauthorized_count = unauthorized_count + ?
    ''', (today, 1 if authorized else 0, 0 if authorized else 1,
          1 if authorized else 0, 0 if authorized else 1))
    
    conn.commit()
    conn.close()


def get_logs(limit=100, offset=0):
    """Get access logs from database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, timestamp, name, role, authorized, confidence, action
        FROM access_logs
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    logs = []
    for row in cursor.fetchall():
        logs.append({
            "id": row[0],
            "timestamp": row[1],
            "name": row[2],
            "role": row[3],
            "authorized": bool(row[4]),
            "confidence": row[5],
            "action": row[6]
        })
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM access_logs')
    total = cursor.fetchone()[0]
    
    conn.close()
    
    return logs, total


def get_daily_stats(days=7):
    """Get daily statistics"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT date, total_access, authorized_count, unauthorized_count
        FROM daily_stats
        ORDER BY date DESC
        LIMIT ?
    ''', (days,))
    
    stats = []
    for row in cursor.fetchall():
        stats.append({
            "date": row[0],
            "total_access": row[1],
            "authorized_count": row[2],
            "unauthorized_count": row[3]
        })
    
    # Get overall stats
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN authorized = 1 THEN 1 ELSE 0 END) as authorized,
            SUM(CASE WHEN authorized = 0 THEN 1 ELSE 0 END) as unauthorized
        FROM access_logs
    ''')
    overall = cursor.fetchone()
    
    conn.close()
    
    return stats, {
        "total_access": overall[0] or 0,
        "authorized_count": overall[1] or 0,
        "unauthorized_count": overall[2] or 0
    }
