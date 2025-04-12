import datetime
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Create a connection to the PostgreSQL database
    
    Returns:
        A connection object to the PostgreSQL database
    """
    try:
        connection = psycopg2.connect(os.environ["DATABASE_URL"])
        return connection
    except psycopg2.Error as e:
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
            id SERIAL PRIMARY KEY,
            filename TEXT,
            result TEXT,
            confidence REAL,
            timestamp TIMESTAMP
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def log_result(filename, result, confidence):
    """
    Log a detection result to the database
    
    Args:
        filename: The filename of the processed image
        result: The detection result ('Real' or 'Fake')
        confidence: Confidence score of the prediction (0-1)
    """
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
    """
    Get recent detection results from the database
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of recent detection results
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute(
            "SELECT * FROM image_log ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving results from database: {e}")
        return []
