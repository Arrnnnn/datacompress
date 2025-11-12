import sqlite3
import os

def analyze_db(db_name):
    try:
        # Use raw string and normalize path
        db_path = os.path.normpath(os.path.join(os.getcwd(), db_name))
        print(f"\n🔍 Analyzing: {db_name}")
        print(f"   Path: {db_path}")
        
        # Check if file exists and is accessible
        if not os.path.exists(db_path):
            print("❌ Error: File does not exist or is not accessible")
            return
            
        # Try to connect
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"   Tables: {', '.join(tables)}")
        
        if 'multimedia_data' in tables:
            # Get item count
            cursor.execute("SELECT COUNT(*) FROM multimedia_data")
            count = cursor.fetchone()[0]
            print(f"   Items in database: {count}")
            
            # Get compression stats
            cursor.execute("""
                SELECT 
                    AVG(compression_ratio) as avg_ratio,
                    SUM(original_size) as total_original,
                    SUM(compressed_size) as total_compressed
                FROM multimedia_data
            """)
            stats = cursor.fetchone()
            
            if stats[0] is not None:  # If there are items
                print(f"   Avg. Compression Ratio: {stats[0]:.2f}x")
                print(f"   Total Original Size: {int(stats[1]):,} bytes")
                print(f"   Total Compressed Size: {int(stats[2]):,} bytes")
                print(f"   Space Saved: {((1 - stats[2]/stats[1])*100):.1f}%" 
                      if stats[1] > 0 else "   Space Saved: N/A")
        
        conn.close()
        print("✅ Analysis complete")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# List of databases to analyze
databases = [
    "interactive_demo.db",
    "multimedia_demo.db",
    "quick_test.db"
]

print("Starting database analysis...")
for db in databases:
    analyze_db(db)
