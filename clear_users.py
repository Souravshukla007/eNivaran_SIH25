import sqlite3
import os

def clear_user_data():
    db_path = 'users.db'
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL statement to delete all records from the 'users' table
        # Replace 'users' with your actual table name if it's different
        sql_query = "DELETE FROM users;"

        # Execute the SQL query
        cursor.execute(sql_query)

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print("All user login details have been successfully deleted.")
        print("The database is now ready for a fresh start.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    clear_user_data()
