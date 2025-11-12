from multimedia_database_example import MultiMediaDatabaseCompressor
import numpy as np

def create_uncompressed_demo():
    # Initialize uncompressed database
    db_path = "uncompressed_demo.db"
    print(f"🔄 Creating uncompressed database: {db_path}")
    
    db = MultiMediaDatabaseCompressor(db_path, compression_level=0)  # 0 = no compression

    # Sample data
    sample_data = {
        "text_data": "This is uncompressed text data",
        "number_list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "numpy_array": np.random.rand(5, 5).tolist(),  # 5x5 random array
        "dictionary": {"name": "Test", "values": [10, 20, 30], "active": True}
    }

    # Store data without compression
    for name, data in sample_data.items():
        db.store_data(name, data, compress=False)
        print(f"✅ Stored: {name}")

    print(f"\n🔍 Verifying database contents...")
    
    # Verify contents
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all items
    cursor.execute("SELECT name, data_type, original_size, compressed_size FROM multimedia_data")
    items = cursor.fetchall()
    
    print("\n📊 Database Contents:")
    print("-" * 50)
    for name, data_type, orig_size, comp_size in items:
        ratio = comp_size / orig_size if orig_size > 0 else 0
        print(f"🔹 {name} ({data_type})")
        print(f"   Original: {orig_size} bytes")
        print(f"   Stored: {comp_size} bytes")
        print(f"   Ratio: {ratio:.2f}x")
        print(f"   {'✅ Uncompressed' if ratio >= 0.95 else '❌ Possibly compressed'}")
        print()
    
    conn.close()
    print(f"\n🎉 Created and verified uncompressed database: {db_path}")

if __name__ == "__main__":
    import sqlite3
    create_uncompressed_demo()
