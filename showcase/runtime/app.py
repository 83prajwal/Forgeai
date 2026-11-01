import json
import os
import re
import sqlite3
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
DB = os.path.join(app.root_path, "showcase.db")
PROFILE = json.loads(open(os.path.join(app.root_path, "profile.json"), encoding="utf-8").read())

def connect():
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    return db

def safe_name(value):
    return re.sub(r"[^a-z0-9_]", "", value.lower())

def init_db():
    with connect() as db:
        for entity in PROFILE["entities"]:
            columns = [f'"{field["name"]}" {"REAL" if field["type"] == "number" else "TEXT"}' for field in entity["fields"]]
            db.execute(f'CREATE TABLE IF NOT EXISTS "{entity["table"]}" (id INTEGER PRIMARY KEY AUTOINCREMENT, {", ".join(columns)}, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
            # Fields added by a later modification become new columns on the existing table.
            existing = {row[1] for row in db.execute(f'PRAGMA table_info("{entity["table"]}")')}
            for field in entity["fields"]:
                if field["name"] not in existing:
                    db.execute(f'ALTER TABLE "{entity["table"]}" ADD COLUMN "{field["name"]}" {"REAL" if field["type"] == "number" else "TEXT"}')
        db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if not db.execute("SELECT 1 FROM settings LIMIT 1").fetchone():
            db.executemany("INSERT INTO settings(key,value) VALUES (?,?)", [("display_name", ""), ("email", ""), ("theme", "light")])

init_db()

@app.get("/")
def index(): return send_from_directory(app.root_path, "index.html")

@app.get("/style.css")
def style(): return send_from_directory(app.root_path, "style.css")

@app.get("/script.js")
def script(): return send_from_directory(app.root_path, "script.js")

@app.get("/api/profile")
def profile(): return jsonify(PROFILE)

@app.get("/api/dashboard")
def dashboard():
    with connect() as db:
        totals = {entity["table"]: db.execute(f'SELECT COUNT(*) FROM "{entity["table"]}"').fetchone()[0] for entity in PROFILE["entities"]}
        activity = []
        for entity in PROFILE["entities"]:
            label = entity["fields"][0]["name"]
            rows = db.execute(f'SELECT "{label}" AS label, created_at FROM "{entity["table"]}" ORDER BY id DESC LIMIT 3').fetchall()
            activity.extend([{ "entity": entity["name"], "label": row["label"], "created_at": row["created_at"] } for row in rows])
    return jsonify({"totals": totals, "activity": activity[:6]})

@app.get("/api/settings")
def get_settings():
    with connect() as db: return jsonify({row["key"]: row["value"] for row in db.execute("SELECT key,value FROM settings")})

@app.put("/api/settings")
def put_settings():
    data = request.get_json(silent=True) or {}
    with connect() as db:
        for key in ("display_name", "email", "theme"):
            if key in data: db.execute("INSERT INTO settings(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(data[key])))
    return get_settings()

def find_entity(slug):
    return next((entity for entity in PROFILE["entities"] if entity["table"] == safe_name(slug)), None)

@app.route("/api/<entity_slug>", methods=["GET", "POST"])
def collection(entity_slug):
    entity = find_entity(entity_slug)
    if not entity: return jsonify({"error":"Unknown collection"}), 404
    if request.method == "GET":
        term = request.args.get("q", "").strip()
        with connect() as db:
            rows = [dict(row) for row in db.execute(f'SELECT * FROM "{entity["table"]}" ORDER BY id DESC')]
        if term: rows = [row for row in rows if term.casefold() in " ".join(str(v) for v in row.values()).casefold()]
        return jsonify({"items":rows})
    data = request.get_json(silent=True) or {}
    fields = [field["name"] for field in entity["fields"]]
    for field in entity["fields"]:
        if field.get("required") and not str(data.get(field["name"], "")).strip(): return jsonify({"error":f'{field["label"]} is required'}), 400
    names = [name for name in fields if name in data]
    if not names: return jsonify({"error":"Enter at least one field"}), 400
    with connect() as db:
        cursor = db.execute(f'INSERT INTO "{entity["table"]}" ({",".join(names)}) VALUES ({",".join("?" for _ in names)})', [data[name] for name in names])
        row = dict(db.execute(f'SELECT * FROM "{entity["table"]}" WHERE id=?', (cursor.lastrowid,)).fetchone())
    return jsonify(row), 201

@app.route("/api/<entity_slug>/<int:item_id>", methods=["GET", "PUT", "DELETE"])
def item(entity_slug, item_id):
    entity = find_entity(entity_slug)
    if not entity: return jsonify({"error":"Unknown collection"}), 404
    with connect() as db:
        if request.method == "GET":
            row = db.execute(f'SELECT * FROM "{entity["table"]}" WHERE id=?', (item_id,)).fetchone()
            if not row: return jsonify({"error":"Not found"}), 404
            return jsonify(dict(row))
        if request.method == "DELETE":
            cursor = db.execute(f'DELETE FROM "{entity["table"]}" WHERE id=?', (item_id,))
            if not cursor.rowcount: return jsonify({"error":"Not found"}), 404
            return jsonify({"deleted":True})
        data = request.get_json(silent=True) or {}
        names = [field["name"] for field in entity["fields"] if field["name"] in data]
        if not names: return jsonify({"error":"No fields to update"}), 400
        db.execute(f'UPDATE "{entity["table"]}" SET {",".join(f"{name}=?" for name in names)} WHERE id=?', [data[name] for name in names]+[item_id])
        row = db.execute(f'SELECT * FROM "{entity["table"]}" WHERE id=?', (item_id,)).fetchone()
        if not row: return jsonify({"error":"Not found"}), 404
        return jsonify(dict(row))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False, use_reloader=False)
