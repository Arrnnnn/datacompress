import os
import sqlite3

def list_and_check_dbs():
    print("🔍 Checking database files in current directory...")
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # List all files in directory
    print("\nFiles in directory:")
    for file in os.listdir(current_dir):
        print(f"- {file}")
    
    # Check .db files
    db_files = [f for f in os.listdir() if f.endswith('.db')]
    print("\nFound database files:")
    for db_file in db_files:
        full_path = os.path.join(current_dir, db_file)
        print(f"\nChecking: {db_file}")
        print(f"Full path: {full_path}")
        
        if not os.path.exists(full_path):
            print("❌ File does not exist!")
            continue
            
        try:
            # Try to open the database
            conn = sqlite3.connect(full_path)
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            print("✅ Database opened successfully")
            print(f"   Tables: {[t[0] for t in tables]}")
            
            # Check if it's a multimedia database
            if any('multimedia_data' in t for t in [t[0] for t in tables]):
                cursor.execute("SELECT COUNT(*) FROM multimedia_data")
                count = cursor.fetchone()[0]
                print(f"   Multimedia items: {count}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    list_and_check_dbs()
