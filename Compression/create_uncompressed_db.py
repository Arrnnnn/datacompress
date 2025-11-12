import sqlite3
import pickle
import numpy as np
from datetime import datetime

def create_simple_uncompressed_db():
    db_path = "simple_uncompressed.db"
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS uncompressed_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        data BLOB NOT NULL,
        data_type TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Sample data
    sample_data = {
        "text_data": "This is uncompressed text data",
        "number_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "numpy_array": np.random.rand(5, 5).tolist(),
        "dictionary": {"name": "Test", "values": [10, 20, 30], "active": True}
    }
    
    # Store data
    for name, data in sample_data.items():
        # Convert data to bytes
        data_bytes = pickle.dumps(data)
        
        # Get data type
        data_type = type(data).__name__
        
        # Insert into database
        cursor.execute('''
            INSERT INTO uncompressed_data (name, data, data_type)
            VALUES (?, ?, ?)
        ''', (name, data_bytes, data_type))
        
        print(f"✅ Stored: {name} ({data_type})")
    
    # Verify data
    print("\n🔍 Verifying stored data:")
    cursor.execute('''
        SELECT name, data_type, LENGTH(data) as size, created_at
        FROM uncompressed_data
    ''')
    
    print("\n{:<15} {:<15} {:<10} {}".format("Name", "Type", "Size", "Created At"))
    print("-" * 50)
    for row in cursor.fetchall():
        name, data_type, size, created_at = row
        print(f"{name:<15} {data_type:<15} {size:<10,} {created_at}")
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Created uncompressed database: {db_path}")

if __name__ == "__main__":
    create_simple_uncompressed_db()
