import sqlite3

def check_table_structure():
    # Connect to the database
    conn = sqlite3.connect('interactive_demo.db')
    cursor = conn.cursor()
    
    # Get table info
    cursor.execute("PRAGMA table_info(multimedia_data)")
    columns = cursor.fetchall()
    
    print("Table structure:")
    for col in columns:
        print(f"- {col[1]} ({col[2]})")
    
    # Show sample data
    print("\nSample data (first row):")
    cursor.execute("SELECT * FROM multimedia_data LIMIT 1")
    row = cursor.fetchone()
    if row:
        for i, value in enumerate(row):
            # Get column name
            col_name = columns[i][1] if i < len(columns) else f"Column {i}"
            # Truncate long values for display
            display_value = str(value)
            if len(display_value) > 100:
                display_value = display_value[:100] + "...[truncated]"
            print(f"{col_name}: {display_value}")
    
    conn.close()

if __name__ == "__main__":
    check_table_structure()
