import sqlite3
import json
from tabulate import tabulate
import os

def view_database_contents(db_path=None):
    # If no path provided, look for database files in current directory
    if db_path is None:
        db_files = [f for f in os.listdir() if f.endswith('.db')]
        if not db_files:
            print("No database files found in current directory")
            return
        db_path = db_files[0]  # Use the first .db file found
        print(f"Using database: {db_path}")

    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nTables in the database:")
        for table in tables:
            print(f"- {table[0]}")
        
        # Show contents of multimedia_data table
        print("\nContents of multimedia_data table:")
        cursor.execute("""
            SELECT 
                id, 
                name, 
                data_type, 
                original_size, 
                compressed_size,
                compression_ratio,
                strftime('%Y-%m-%d %H:%M:%S', created_at) as created
            FROM multimedia_data
            ORDER BY id
        """)
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        # Display as a nice table
        print(tabulate(rows, headers=columns, tablefmt='grid', numalign='right', stralign='left'))
        
        # Show detailed info for each item
        cursor.execute("SELECT id, name, data_type, metadata, processing_info FROM multimedia_data")
        items = cursor.fetchall()
        
        for item in items:
            item_id, name, data_type, metadata, processing_info = item
            print(f"\n📁 Item: {name} (ID: {item_id})")
            print(f"   Type: {data_type}")
            
            if metadata:
                try:
                    meta = json.loads(metadata)
                    print("   Metadata:")
                    for key, value in meta.items():
                        print(f"     {key}: {value}")
                except:
                    print(f"   Metadata: {metadata}")
            
            if processing_info:
                try:
                    proc_info = json.loads(processing_info)
                    print("   Processing Info:")
                    for key, value in proc_info.items():
                        print(f"     {key}: {value}")
                except:
                    print(f"   Processing Info: {processing_info}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    view_database_contents()
