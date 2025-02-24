
# database_manager.py
import sqlite3
from threading import Lock
from werkzeug.security import generate_password_hash, check_password_hash

class SecureDatabase:
    """Thread-safe SQLite database manager"""
    def __init__(self, db_path: str = "ai_system.db"):
        self.db_path = db_path
        self.lock = Lock()
        self._init_db()

    def _init_db(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    query TEXT,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )""")

    def create_user(self, username: str, password: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, generate_password_hash(password)))

    def authenticate(self, username: str, password: str) -> bool:
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            return result and check_password_hash(result[0], password)
