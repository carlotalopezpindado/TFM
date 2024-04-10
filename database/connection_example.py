import mysql.connector
from mysql.connector import Error

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

host = config['database']['host']
database = config['database']['database']
user = config['database']['user']
password = config['database']['password']

try:
    # Connect to the database
    connection = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    
    if connection.is_connected():
        db_info = connection.get_server_info()
        print("Successfully connected to MySQL Server version ", db_info)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM tfm.users")

        records = cursor.fetchall()
        print("Total number of users: ", cursor.rowcount)
        
        print("\nList of users:")
        for row in records:
            print("Id = ", row[0], "UserName = ", row[1], "Rol = ", row[2])

except Error as e:
    print("Error while connecting to MySQL", e)

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
