from compression_pipeline.pipeline import CompressionPipeline
from compression_pipeline.models import CompressedData
from multimedia_database_example import MultiMediaDatabaseCompressor
import sqlite3
import pickle
import numpy as np

class BetterCompressor:
    def __init__(self):
        self.db_path = "optimized_compressed.db"
        self.conn = sqlite3.connect(self.db_path)
        self.setup_database()
        self.pipeline = CompressionPipeline(quality=0.95, block_size=8)

    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compressed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                data_type TEXT NOT NULL,
                original_size INTEGER NOT NULL,
                compressed_size INTEGER NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        self.conn.commit()

    def ensure_2d_array(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim == 1:
            return data.reshape(-1, 1)
        return data

    def prepare_data(self, data, data_type):
        try:
            if data_type == 'str':
                return np.array([ord(c) for c in data], dtype=np.uint8).reshape(-1, 1)
            elif data_type == 'list':
                arr = np.array(data, dtype=np.float64)
                return arr.reshape(-1, 1) if arr.ndim == 1 else arr
            elif data_type == 'dict':
                import json
                json_str = json.dumps(data, sort_keys=True)
                return np.array([ord(c) for c in json_str], dtype=np.uint8).reshape(-1, 1)
            return self.ensure_2d_array(data)
        except Exception as e:
            print(f"   ⚠️ Data preparation error: {str(e)}")
            return None

    def store_compressed_data(self, name, data, data_type):
        try:
            # Compress the data
            compressed_obj = self.pipeline.compress(data)
            compressed_bytes = compressed_obj.tobytes() if hasattr(compressed_obj, 'tobytes') else pickle.dumps(compressed_obj)
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO compressed_data 
                (name, data_type, original_size, compressed_size, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                data_type,
                data.nbytes,
                len(compressed_bytes),
                sqlite3.Binary(compressed_bytes)
            ))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"   ❌ Storage error: {str(e)}")
            self.conn.rollback()
            return False

    def run(self):
        conn = sqlite3.connect('simple_uncompressed.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, data, data_type FROM uncompressed_data")
        
        for name, data_bytes, data_type in cursor.fetchall():
            try:
                original_data = pickle.loads(data_bytes)
                print(f"\n📦 Processing: {name} ({data_type})")
                
                prepared_data = self.prepare_data(original_data, data_type)
                if prepared_data is None:
                    print("   ⏩ Skipping - preparation failed")
                    continue
                
                print(f"   Data shape: {prepared_data.shape}")
                print(f"   Data type: {prepared_data.dtype}")
                
                # Store the compressed data
                if self.store_compressed_data(name, prepared_data, data_type):
                    print("   ✅ Successfully stored compressed data")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        conn.close()
        self.conn.close()

if __name__ == "__main__":
    compressor = BetterCompressor()
    compressor.run()
