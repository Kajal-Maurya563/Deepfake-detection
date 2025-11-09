import datetime
import os
import logging
import mysql.connector
from mysql.connector import Error

logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
        )
        return connection
    except Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def init_db():
    """Initialize the database with required tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create the table for logging detection results
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename TEXT,
            result TEXT,
            confidence FLOAT,
            timestamp DATETIME
        )
        ''')

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def log_result(filename, result, confidence):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        timestamp = datetime.datetime.now()

        cursor.execute(
            "INSERT INTO image_log (filename, result, confidence, timestamp) VALUES (%s, %s, %s, %s)",
            (filename, result, confidence, timestamp)
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.debug(f"Result logged: {filename}, {result}, {confidence:.2f}")
    except Exception as e:
        logger.error(f"Error logging result to database: {e}")

def get_recent_results(limit=10):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            f"SELECT * FROM image_log ORDER BY timestamp DESC LIMIT {limit}"
        )

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return results
    except Exception as e:
        logger.error(f"Error retrieving results from database: {e}")
        return []
