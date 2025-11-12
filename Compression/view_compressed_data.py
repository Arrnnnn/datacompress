import sqlite3
import numpy as np
import pickle

def view_compressed_data(db_path="optimized_compressed.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all data
    cursor.execute("""
        SELECT name, data_type, original_size, compressed_size, data 
        FROM compressed_data
        ORDER BY name
    """)
    
    print("\n" + "="*80)
    print("COMPRESSION REPORT")
    print("="*80)
    
    total_original = 0
    total_compressed = 0
    
    for row in cursor.fetchall():
        name, data_type, orig_size, comp_size, data = row
        ratio = orig_size / comp_size if comp_size > 0 else 0
        space_saved = (1 - (comp_size / orig_size)) * 100 if orig_size > 0 else 0
        
        print(f"\n📊 {name.upper()}")
        print("-" * 40)
        print(f"Type: {data_type}")
        print(f"Original size: {orig_size:,} bytes")
        print(f"Compressed size: {comp_size:,} bytes")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Space saved: {space_saved:+.1f}%")
        
        # Show preview of the data if it's text
        if data_type in ['str', 'dict']:
            try:
                # Try to decode as text
                text = data.decode('utf-8', errors='replace')
                print(f"\nPreview: {text[:100]}..." if len(text) > 100 else f"\nData: {text}")
            except:
                print("\n[Binary data - cannot display]")
        
        total_original += orig_size
        total_compressed += comp_size
    
    # Print summary
    total_ratio = total_original / total_compressed if total_compressed > 0 else 0
    total_space_saved = (1 - (total_compressed / total_original)) * 100 if total_original > 0 else 0
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total original size: {total_original:,} bytes")
    print(f"Total compressed size: {total_compressed:,} bytes")
    print(f"Overall compression ratio: {total_ratio:.2f}x")
    print(f"Total space saved: {total_space_saved:+.1f}%")
    print("="*80)
    
    conn.close()

if __name__ == "__main__":
    view_compressed_data()
