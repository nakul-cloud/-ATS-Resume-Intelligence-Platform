import pyodbc

SQL_SERVER = r"NAKUL-16\SQLEXPRESS"
SQL_DATABASE = "ResumeATS"

def get_conn():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        "Trusted_Connection=yes;"
    )