import sqlite3, datetime

def log_result(filename, result):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS image_log
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, result TEXT, timestamp TEXT)''')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO image_log (filename, result, timestamp) VALUES (?, ?, ?)",
              (filename, result, timestamp))
    conn.commit()
    conn.close()
