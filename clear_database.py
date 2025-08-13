import sqlite3
import os

# --- Configuration ---
# Ensure this path points to your application's database
APP_DB = os.path.join(os.path.dirname(__file__), 'enivaran.db')

def clear_all_data():
    """
    Connects to the database and erases all data from the complaints,
    users, upvotes, and pothole_detections tables. It also resets the
    auto-increment counters for these tables.
    """
    if not os.path.exists(APP_DB):
        print(f"Error: Database file not found at '{APP_DB}'")
        return

    try:
        with sqlite3.connect(APP_DB) as conn:
            cursor = conn.cursor()
            print("Connected to the database...")

            # Temporarily disable foreign key constraints to allow table clearing
            cursor.execute('PRAGMA foreign_keys = OFF')
            print("Temporarily disabled foreign key constraints.")

            # List of tables to clear
            tables_to_clear = ['complaints', 'users', 'upvotes', 'pothole_detections']
            
            for table in tables_to_clear:
                # Check if table exists before trying to delete from it
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    print(f"Clearing data from '{table}' table...")
                    cursor.execute(f'DELETE FROM {table}')
                    # Reset the auto-increment counter for the table
                    cursor.execute(f"DELETE FROM sqlite_sequence WHERE name='{table}'")
                    print(f"Data and auto-increment counter for '{table}' have been reset.")
                else:
                    print(f"Table '{table}' not found, skipping.")

            # Re-enable foreign key constraints
            cursor.execute('PRAGMA foreign_keys = ON')
            print("Re-enabled foreign key constraints.")

            # Commit the changes
            conn.commit()
            print("\nDatabase has been successfully cleared.")

    except sqlite3.Error as e:
        print(f"\nAn error occurred: {e}")
        # The 'with' statement will automatically handle rollback on error

if __name__ == '__main__':
    # Confirmation prompt to prevent accidental execution
    confirm = input("Are you absolutely sure you want to erase all user and complaint data? This action is irreversible. (yes/no): ")
    if confirm.lower() == 'yes':
        clear_all_data()
    else:
        print("Operation cancelled.")
