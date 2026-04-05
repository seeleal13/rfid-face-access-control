
from flask import Flask, request, jsonify
import os
import shutil

app = Flask(__name__)

DB_DEST = "faces.db"        # Path where DB will be saved on the Pi
TEMP_PATH = "faces.tmp"     # Temp file to avoid corrupt writes

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    
    if not f:
        return jsonify({"error": "Aucun fichier reÃ§u"}), 400

    # Save to temp file first, then replace atomically
    f.save(TEMP_PATH)
    shutil.move(TEMP_PATH, DB_DEST)
    
    print(f"[âœ”] DB reÃ§ue et mise Ã  jour ! {DB_DEST}")
    return jsonify({"status": "ok"}), 200

@app.route("/status", methods=["GET"])
def status():
    exists = os.path.exists(DB_DEST)
    size   = os.path.getsize(DB_DEST) if exists else 0
    return jsonify({
        "db_present": exists,
        "db_size_bytes": size
    }), 200

if __name__ == "__main__":
    print("ðŸš€ Serveur Flask dÃ©marrÃ© sur le port 5000...")
    print(f"   DB destination : {os.path.abspath(DB_DEST)}")
    app.run(host="192.168.1.11", port=5000, debug=False)