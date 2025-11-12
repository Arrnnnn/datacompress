import sqlite3
import json
import os
from datetime import datetime

def extract_video(output_dir="extracted_videos"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect('interactive_demo.db')
    cursor = conn.cursor()
    
    try:
        # Get video data
        cursor.execute("""
            SELECT name, compressed_blob, processing_info, file_extension,
                   original_size, compressed_size, compression_ratio
            FROM multimedia_data 
            WHERE data_type = 'video'
        """)
        
        for row in cursor.fetchall():
            # Unpack the row using numeric indices
            name = row[0]
            video_data = row[1]
            processing_info = json.loads(row[2])
            file_extension = row[3] or '.mp4'
            original_size = row[4]
            compressed_size = row[5]
            compression_ratio = row[6]
            
            # Create output filename
            output_filename = f"extracted_{name}{file_extension}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the binary data
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            print(f"\n✅ Video extracted to: {output_path}")
            print(f"   Resolution: {processing_info.get('width', 'N/A')}x{processing_info.get('height', 'N/A')}")
            print(f"   Frames: {processing_info.get('frames', 'N/A')}")
            print(f"   FPS: {processing_info.get('fps', 'N/A')}")
            print(f"   Original size: {original_size:,} bytes")
            print(f"   Compressed size: {compressed_size:,} bytes")
            print(f"   Compression ratio: {compression_ratio:.2f}x")
    
    except Exception as e:
        print(f"\n❌ Error extracting video: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    print("=== Extracting Videos ===")
    extract_video()
