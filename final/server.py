from flask import Flask, jsonify, request
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
DB_PATH = "gesture_log.db"
current_gesture = "none"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS gesture_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@app.route("/update", methods=["POST"])
def update_gesture():
    global current_gesture
    data = request.get_json()
    if data and "gesture" in data:
        current_gesture = data["gesture"]
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO gesture_log (gesture, timestamp) VALUES (?, ?)",
            (current_gesture, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "gesture": current_gesture})
    return jsonify({"status": "error", "message": "missing gesture"}), 400


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"gesture": current_gesture})


@app.route("/logs", methods=["GET"])
def get_logs():
    limit = request.args.get("limit", 50, type=int)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT gesture, timestamp FROM gesture_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return jsonify([{"gesture": r[0], "timestamp": r[1]} for r in rows])


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
