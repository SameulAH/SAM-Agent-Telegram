"""Purge poisoned STM history."""
import sqlite3
conn = sqlite3.connect("/app/data/memory.db")
c = conn.cursor()

# Find tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f"Tables: {tables}")

# Delete all conversation history for this chat
for table in tables:
    c.execute(f"DELETE FROM {table} WHERE conversation_id = ?", ("telegram_903341171",))
    print(f"  Deleted {c.rowcount} rows from {table}")

conn.commit()
conn.close()
print("Done - poisoned history purged")
