import cv2
import mediapipe as mp
import time
import numpy as np
import onnxruntime as ort
import sqlite3
import requests
import threading
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DB_PATH        = "faces.db"
MODEL_ONNX     = "w600k_mbf.onnx"
LANDMARK_MODEL = "face_landmarker.task"
RPI_URL        = "http://192.168.1.11:5000/upload"

N_FRAMES       = 60   # Frames pour stabiliser une capture
N_CAPTURES     = 10   # Nombre de captures pour la moyenne finale
STAB_SEUIL     = 0.8  # Seuil de stabilité (plus bas = plus strict)

# EAR calibration
MIN_DROP           = 0.12
THRESHOLD_MIN      = 0.07
THRESHOLD_MAX      = 0.7
MAX_CALIB_RETRIES  = 2

# indices des 16 points de l'oeil gauche
INDICES_G = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# indices des 16 points de l'oeil droit
INDICES_D = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 5 points pour l'alignement
dst_pts = np.array([
    [38.29, 51.69], [73.53, 51.50], [56.02, 71.73],
    [41.54, 92.36], [70.72, 92.20]
], dtype=np.float32)


def db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personnes (
            card_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            ear_threshold REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def db_upsert(card_id, full_name, embedding_512, ear_threshold):
    conn = sqlite3.connect(DB_PATH)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blob = embedding_512.astype(np.float32).tobytes()
    conn.execute("""
        INSERT INTO personnes (card_id, full_name, embedding, ear_threshold, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(card_id) DO UPDATE SET
            full_name=excluded.full_name,
            embedding=excluded.embedding,
            ear_threshold=excluded.ear_threshold,
            timestamp=excluded.timestamp
    """, (card_id, full_name, blob, float(ear_threshold), ts))
    conn.commit()
    conn.close()


def send_db_to_pi():
    try:
        with open(DB_PATH, "rb") as f:
            response = requests.post(
                RPI_URL,
                files={"file": ("faces.db", f, "application/octet-stream")},
                timeout=10
            )
        if response.status_code == 200:
            print("DB synchronisée avec le Raspberry Pi.")
        else:
            print(f"Pi a répondu avec le code {response.status_code}.")
    except requests.exceptions.ConnectionError:
        print("Impossible de joindre le Raspberry Pi.")
    except requests.exceptions.Timeout:
        print("Timeout — le Raspberry Pi n'a pas répondu.")
    except Exception as e:
        print(f"Erreur envoi DB : {e}")


session  = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

def get_embedding(face_bgr_112):
    rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
    f   = (rgb.astype(np.float32) - 127.5) / 128.0
    inp = np.expand_dims(np.transpose(f, (2, 0, 1)), 0)
    emb = session.run([out_name], {inp_name: inp})[0][0]
    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 1e-6 else emb.astype(np.float32)


dst_pts = np.array([
    [38.29, 51.69], [73.53, 51.50], [56.02, 71.73],
    [41.54, 92.36], [70.72, 92.20]
], dtype=np.float32)

def get_landmark_pts(lm, w, h):
    return np.array([
        [(lm[33].x + lm[133].x) * w / 2, (lm[33].y + lm[133].y) * h / 2],
        [(lm[362].x + lm[263].x) * w / 2, (lm[362].y + lm[263].y) * h / 2],
        [lm[1].x  * w, lm[1].y  * h],
        [lm[61].x * w, lm[61].y * h],
        [lm[291].x * w, lm[291].y * h]
    ], dtype=np.float32)

def is_face_aligned(lm):
    eye_dist    = abs(lm[263].x - lm[33].x)
    nose_offset = abs(lm[1].x - (lm[33].x + lm[263].x) / 2)
    return nose_offset < (eye_dist * 0.15)

def calculate_ear(points_g, points_d):
    A_g = np.linalg.norm(points_g[4] - points_g[12])
    B_g = np.linalg.norm(points_g[5] - points_g[11])
    C_g = np.linalg.norm(points_g[3] - points_g[13])
    D_g = np.linalg.norm(points_g[6] - points_g[10])
    E_g = np.linalg.norm(points_g[2] - points_g[14])
    F_g = np.linalg.norm(points_g[0] - points_g[3])
    ear_g = (A_g + B_g + C_g + D_g + E_g) / (5.0 * F_g) if F_g > 1e-6 else 0.0

    A_d = np.linalg.norm(points_d[4] - points_d[12])
    B_d = np.linalg.norm(points_d[5] - points_d[11])
    C_d = np.linalg.norm(points_d[3] - points_d[13])
    D_d = np.linalg.norm(points_d[6] - points_d[10])
    E_d = np.linalg.norm(points_d[2] - points_d[14])
    F_d = np.linalg.norm(points_d[0] - points_d[3])
    ear_d = (A_d + B_d + C_d + D_d + E_d) / (5.0 * F_d) if F_d > 1e-6 else 0.0

    return ear_g, ear_d, (ear_g + ear_d) / 2.0

class PassiveCalibrator:
    def __init__(self, n_needed: int = 3):
        self.threshold   = None
        self.calibrated  = False
        self.not_real    = False
        self.n_needed    = n_needed
        self.retry_count = 0
        self._blinks     = []
        self._baseline   = None
        self._ear_min    = 0.0
        self._in_blink   = False
        self._last_t     = 0.0

    def update(self, ear: float, now: float):
        if self.calibrated or self.not_real:
            return

        if self._baseline is None:
            self._baseline = ear
            return

        self._baseline = 0.97 * self._baseline + 0.03 * ear

        if not self._in_blink:
            if self._baseline - ear >= MIN_DROP:
                self._in_blink = True
                self._ear_min  = ear
        else:
            self._ear_min = min(self._ear_min, ear)

            if ear >= self._baseline - MIN_DROP * 0.3:
                self._in_blink = False

                if now - self._last_t < 0.4:
                    return

                self._last_t = now
                self._blinks.append((self._ear_min, self._baseline))
                n = len(self._blinks)
                print(f"[Calibration] blink {n}/{self.n_needed}  ear_min={self._ear_min:.4f}")

                if n >= self.n_needed:
                    candidate = round(max(0.5 * (eo + 2 * em) for em, eo in self._blinks), 4)

                    if THRESHOLD_MIN <= candidate <= THRESHOLD_MAX:
                        self.threshold  = candidate
                        self.calibrated = True
                        print(f"[Calibration] seuil EAR = {candidate:.4f}")
                    else:
                        self.retry_count += 1
                        print(f"[Calibration] seuil rejeté = {candidate:.4f}")
                        self._blinks = []
                        self._last_t = 0.0
                        if self.retry_count >= MAX_CALIB_RETRIES:
                            self.not_real = True
                            print("Pas une personne réelle.")


latest_lm = None
lm_lock = threading.Lock()

def mediapipe_callback(result, *_):
    global latest_lm
    with lm_lock:
        latest_lm = result.face_landmarks[0] if result.face_landmarks else None

def start_enrollment(detector):
    global latest_lm
    print("\n--- NOUVEL ENRÔLEMENT ---")
    prenom = input("Prénom : ").strip()
    nom    = input("Nom : ").strip()
    card_id = input("card_id : ").strip()
    full_name = f"{prenom} {nom}".strip()

    if not prenom or not nom or not card_id:
        print("Erreur: Champs vides.")
        return

    cap = cv2.VideoCapture(0)
    emb_list, aligned_pts, saved = [], [], 0
    calibrator = PassiveCalibrator()
    ear_threshold = None

    print(f"Alignez votre visage dans le rectangle VERT pour {full_name}...")
    print("Puis clignez des yeux pour calibrer l'EAR.")

    while saved < N_CAPTURES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        sq    = cv2.resize(frame[:, (w - h) // 2 : (w - h) // 2 + h], (480, 480))

        now = time.perf_counter()
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        )
        detector.detect_async(mp_img, int(time.time() * 1000))

        color = (0, 0, 255)

        with lm_lock:
            lm = latest_lm

        if lm:
            if is_face_aligned(lm):
                color = (0, 255, 0)

                pts_g = np.array([(lm[i].x * 480, lm[i].y * 480) for i in INDICES_G], dtype=np.float32)
                pts_d = np.array([(lm[i].x * 480, lm[i].y * 480) for i in INDICES_D], dtype=np.float32)
                _, _, ear = calculate_ear(pts_g, pts_d)

                if not calibrator.calibrated and not calibrator.not_real:
                    calibrator.update(ear, now)

                if calibrator.calibrated:
                    ear_threshold = calibrator.threshold

                    aligned_pts.append(get_landmark_pts(lm, 480, 480))

                    if len(aligned_pts) >= N_FRAMES:
                        pts_mean  = np.mean(aligned_pts, axis=0)
                        std_score = np.mean(np.std(aligned_pts, axis=0))

                        if std_score < STAB_SEUIL:
                            M, _ = cv2.estimateAffinePartial2D(pts_mean, dst_pts)
                            if M is not None:
                                face = cv2.warpAffine(sq, M, (112, 112))
                                emb_list.append(get_embedding(face))
                                saved += 1
                                print(f"Capture {saved}/{N_CAPTURES} validée")
                        aligned_pts = []

                xs = [l.x for l in lm]
                ys = [l.y for l in lm]
                cv2.rectangle(
                    sq,
                    (int(min(xs) * 480), int(min(ys) * 480)),
                    (int(max(xs) * 480), int(max(ys) * 480)),
                    color, 2
                )

        if calibrator.calibrated:
            txt = f"EAR calibré: {calibrator.threshold:.4f} | Captures: {saved}/{N_CAPTURES}"
        elif calibrator.not_real:
            txt = "Calibration échouée"
        else:
            txt = "Clignez des yeux pour calibrer l'EAR"

        cv2.putText(sq, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imshow("Moniteur Alignement", sq)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved >= N_CAPTURES and ear_threshold is not None:
        final_embedding = np.mean(np.stack(emb_list), axis=0)
        final_embedding /= np.linalg.norm(final_embedding)
        db_upsert(card_id, full_name, final_embedding, ear_threshold)
        print(f"SUCCÈS: {full_name} (card_id: {card_id}) enregistré avec EAR={ear_threshold:.4f}")
        send_db_to_pi()


if __name__ == "__main__":
    db_init()
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARK_MODEL),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=mediapipe_callback
    )

    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        while True:
            start_enrollment(landmarker)
            if input("\nContinuer ? (o/n) : ").lower() != 'o':
                break