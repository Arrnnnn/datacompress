# Save this as better_compressor.py
import sqlite3
import numpy as np
import pickle
import os
import zlib
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from compression_pipeline import CompressionPipeline

class SmartDatabaseCompressor:
    def __init__(self, input_db_path: str, output_db_path: str = None, 
                 quality: float = 0.9, block_size: int = 8):
        self.input_db = input_db_path
        self.output_db = output_db_path or f"compressed_{os.path.basename(input_db_path)}"
        self.quality = quality
        self.block_size = block_size
        self.stats = {
            'total_original': 0,
            'total_compressed': 0,
            'tables_processed': 0,
            'compression_ratios': [],
            'skipped_columns': []
        }
        
    def _is_already_compressed(self, data: bytes) -> bool:
        """Check if data is already compressed"""
        if not isinstance(data, bytes) or len(data) < 2:
            return False
        
        # Check for common compression headers
        if data.startswith(b'\x1f\x8b'):  # gzip
            return True
        if data.startswith((b'x\x01', b'x\x9c', b'x\xda')):  # zlib
            return True
        if data.startswith((b'BZh', b'PK\x03\x04', b'Rar!', b'7z\xbc\xaf')):  # bz2, zip, rar, 7z
            return True
            
        return False

    def _compress_data(self, data: Any) -> Tuple[bytes, str, float]:
        """Compress data using appropriate method based on type"""
        try:
            # Skip None or empty data
            if data is None:
                return None, 'null', 1.0
                
            # Convert to bytes if not already
            if not isinstance(data, bytes):
                try:
                    data = pickle.dumps(data)
                except Exception as e:
                    print(f"   ⚠️ Could not serialize data: {str(e)}")
                    return None, 'error', 1.0
            
            # Skip if already compressed
            if self._is_already_compressed(data):
                return data, 'already_compressed', 1.0
                
            # Try different compression methods
            try:
                # First try zlib (good balance of speed and ratio)
                compressed = zlib.compress(data, level=6)
                if len(compressed) < len(data) * 0.9:  # Only use if significant saving
                    return compressed, 'zlib', len(data) / len(compressed)
            except:
                pass
                
            # Fall back to custom pipeline for numeric data
            try:
                # Try to convert to numpy array
                arr = np.array(data)
                if np.issubdtype(arr.dtype, np.number):  # Only for numeric data
                    pipeline = CompressionPipeline(quality=self.quality, block_size=self.block_size)
                    compressed = pipeline.compress(arr)
                    if hasattr(compressed, 'tobytes'):
                        compressed = compressed.tobytes()
                    return compressed, 'custom', len(data) / len(compressed)
            except:
                pass
                
            # If no better option, return original
            return data, 'uncompressed', 1.0
            
        except Exception as e:
            print(f"   ⚠️ Compression error: {str(e)}")
            return data, 'error', 1.0

    def process_database(self):
        """Process the database with smart compression"""
        try:
            # Connect to source database
            src_conn = sqlite3.connect(f'file:{self.input_db}?immutable=1', uri=True)
            src_conn.row_factory = sqlite3.Row
            src_cur = src_conn.cursor()
            
            # Create output database
            if os.path.exists(self.output_db):
                os.remove(self.output_db)
            dest_conn = sqlite3.connect(self.output_db)
            dest_cur = dest_conn.cursor()
            
            # Create metadata table
            dest_cur.execute('''
                CREATE TABLE compression_stats (
                    table_name TEXT,
                    column_name TEXT,
                    original_size INTEGER,
                    compressed_size INTEGER,
                    compression_ratio REAL,
                    compression_type TEXT,
                    row_count INTEGER,
                    PRIMARY KEY (table_name, column_name)
                )
            ''')
            
            # Get all tables
            src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in src_cur.fetchall() 
                     if not row[0].startswith('sqlite_') and row[0] != 'compression_stats']
            
            for table in tables:
                print(f"\n📊 Processing table: {table}")
                
                # Get table structure
                src_cur.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in src_cur.fetchall()]
                
                # Create compressed table
                dest_cur.execute(f'''
                    CREATE TABLE {table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_data BLOB,
                        compressed_data BLOB,
                        compression_ratio REAL,
                        compression_type TEXT
                    )
                ''')
                
                # Process each row
                src_cur.execute(f"SELECT * FROM {table}")
                rows_processed = 0
                
                for row in src_cur:
                    row_data = {}
                    for col in columns:
                        try:
                            data = row[col]
                            if data is None:
                                row_data[col] = (None, 'null', 1.0)
                                continue
                                
                            # Compress the data
                            compressed_data, comp_type, ratio = self._compress_data(data)
                            
                            # Update stats
                            orig_size = len(pickle.dumps(data)) if not isinstance(data, bytes) else len(data)
                            comp_size = len(compressed_data) if compressed_data is not None else 0
                            
                            self.stats['total_original'] += orig_size
                            self.stats['total_compressed'] += comp_size
                            self.stats['compression_ratios'].append(ratio)
                            
                            row_data[col] = (compressed_data, comp_type, ratio)
                            
                        except Exception as e:
                            print(f"   ⚠️ Error processing {table}.{col}: {str(e)}")
                            self.stats['skipped_columns'].append(f"{table}.{col}")
                            row_data[col] = (None, 'error', 1.0)
                    
                    # Store the row
                    try:
                        # Store as JSON for simplicity
                        row_json = json.dumps({k: v[0] for k, v in row_data.items() if v[0] is not None})
                        compressed_row, comp_type, ratio = self._compress_data(row_json)
                        
                        dest_cur.execute(f'''
                            INSERT INTO {table} 
                            (original_data, compressed_data, compression_ratio, compression_type)
                            VALUES (?, ?, ?, ?)
                        ''', (
                            sqlite3.Binary(row_json.encode('utf-8')),
                            sqlite3.Binary(compressed_row) if compressed_row else None,
                            ratio,
                            comp_type
                        ))
                        
                        rows_processed += 1
                        if rows_processed % 100 == 0:
                            print(f"   Processed {rows_processed} rows...")
                            
                    except Exception as e:
                        print(f"   ⚠️ Error storing row: {str(e)}")
                
                # Update stats
                dest_cur.execute('''
                    INSERT INTO compression_stats 
                    (table_name, column_name, original_size, compressed_size, 
                     compression_ratio, compression_type, row_count)
                    SELECT 
                        ? as table_name,
                        'all_columns' as column_name,
                        SUM(LENGTH(original_data)) as original_size,
                        SUM(LENGTH(compressed_data)) as compressed_size,
                        AVG(compression_ratio) as compression_ratio,
                        'overall' as compression_type,
                        COUNT(*) as row_count
                    FROM {}
                '''.format(table), (table,))
                
                self.stats['tables_processed'] += 1
                dest_conn.commit()
                print(f"   ✅ Completed {table} ({rows_processed} rows)")
            
            # Save final summary
            self._save_final_stats(dest_cur)
            dest_conn.commit()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            src_conn.close()
            if 'dest_conn' in locals():
                dest_conn.close()

    def _save_final_stats(self, cursor):
        """Save final compression statistics"""
        if not self.stats['compression_ratios']:
            return
            
        total_ratio = (self.stats['total_original'] / self.stats['total_compressed'] 
                      if self.stats['total_compressed'] > 0 else 0)
        space_saved = (1 - (self.stats['total_compressed'] / self.stats['total_original'])) * 100
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compression_summary (
                total_original_size INTEGER,
                total_compressed_size INTEGER,
                average_compression_ratio REAL,
                space_saved_percent REAL,
                tables_processed INTEGER,
                skipped_columns TEXT
            )
        ''')
        
        cursor.execute('DELETE FROM compression_summary')
        cursor.execute('''
            INSERT INTO compression_summary VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.stats['total_original'],
            self.stats['total_compressed'],
            total_ratio,
            space_saved,
            self.stats['tables_processed'],
            json.dumps(self.stats['skipped_columns'])
        ))

    def print_stats(self):
        """Print compression statistics"""
        if not self.stats['compression_ratios']:
            print("No compression statistics available.")
            return
            
        total_ratio = (self.stats['total_original'] / self.stats['total_compressed'] 
                      if self.stats['total_compressed'] > 0 else 0)
        space_saved = (1 - (self.stats['total_compressed'] / self.stats['total_original'])) * 100
        
        print("\n" + "="*80)
        print("COMPRESSION SUMMARY")
        print("="*80)
        print(f"Source database: {self.input_db}")
        print(f"Compressed database: {self.output_db}")
        print(f"Tables processed: {self.stats['tables_processed']}")
        print(f"Total original size: {self.stats['total_original']:,} bytes")
        print(f"Total compressed size: {self.stats['total_compressed']:,} bytes")
        print(f"Overall compression ratio: {total_ratio:.2f}x")
        print(f"Space saved: {space_saved:+.1f}%")
        
        if self.stats['skipped_columns']:
            print(f"\n⚠️  Skipped columns (already compressed or uncompressible):")
            for col in self.stats['skipped_columns'][:5]:  # Show first 5
                print(f"   - {col}")
            if len(self.stats['skipped_columns']) > 5:
                print(f"   ... and {len(self.stats['skipped_columns']) - 5} more")
        
        print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart SQLite Database Compressor')
    parser.add_argument('input_db', help='Path to the input SQLite database')
    parser.add_argument('-o', '--output', help='Output database path (optional)')
    parser.add_argument('-q', '--quality', type=float, default=0.9,
                       help='Compression quality (0.1-1.0, default: 0.9)')
    parser.add_argument('-b', '--block-size', type=int, default=8, choices=[4, 8, 16],
                       help='Compression block size (default: 8)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting smart compression of {args.input_db}")
    print(f"🔧 Settings: Quality={args.quality}, Block Size={args.block_size}")
    
    compressor = SmartDatabaseCompressor(
        input_db_path=args.input_db,
        output_db_path=args.output,
        quality=args.quality,
        block_size=args.block_size
    )
    
    if compressor.process_database():
        compressor.print_stats()
        print(f"\n✅ Compression completed successfully!")
        print(f"   Output database: {compressor.output_db}")
    else:
        print("\n❌ Compression failed. Check error messages above.")

if __name__ == "__main__":
    main()
