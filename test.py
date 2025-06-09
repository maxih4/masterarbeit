import psycopg

# Beispiel für psycopg2, falls du diese Version nutzt
connection_string = "postgresql://postgres:example@localhost:5432/postgres?sslmode=disable"

try:
    conn = psycopg.connect(connection_string)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print("Verbindung erfolgreich!")
    print(cur.fetchone())
    cur.close()
    conn.close()
except Exception as e:
    print("Fehler bei der Verbindung:", e)