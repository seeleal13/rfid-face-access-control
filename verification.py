
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import sqlite3
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


DB_PATH        = "faces.db"
MODEL_ONNX     = "w600k_mbf.onnx"
LANDMARK_MODEL = "face_landmarker.task"
JSON_KEYFILE   = "projet-pfe-491720-01ddef8b040c.json"
SHEET_ID       = "1_vtU4zNF_jHIilncXn4QcS9uPqf_w97GU_etPVPnq-g"

PC_STREAM_URL  = "http://192.168.1.10:5000/video"

THRESHOLD       = 0.5
N_FRAMES        = 20
STAB_SEUIL      = 0.8
RESULT_DURATION = 2.0

last_event: dict = {}


YAW_MAX            = 15
PITCH_MAX          = 15
MAX_BLINK_INTERVAL = 8
NO_EAR_MAX_FRAMES  = 30
EAR_SMOOTH_FRAMES  = 2
LABEL_STABLE_N     = 12
LIVENESS_TIMEOUT   = 30

INDICES_G    = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
INDICES_D    = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
POSE_INDICES = [4, 152, 33, 263, 61, 291]
FONT_V       = cv2.FONT_HERSHEY_SIMPLEX

MODEL_POINTS_3D = np.array([
    ( 0.0,    0.0,    0.0),
    ( 0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1))


def db_get_person(card_id):
    """Retourne (full_name, embedding, ear_threshold) depuis faces.db local."""
    conn = sqlite3.connect(DB_PATH)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(personnes)").fetchall()}
    col     = "nom_complet" if "nom_complet" in cols else "full_name"
    has_ear = "ear_threshold" in cols
    select  = f"{col}, embedding, ear_threshold" if has_ear else f"{col}, embedding"
    row     = conn.execute(
        f"SELECT {select} FROM personnes WHERE card_id=?", (card_id,)
    ).fetchone()
    conn.close()

    if row is None:
        return None, None, None

    nom           = row[0]
    emb           = np.frombuffer(row[1], dtype=np.float32).copy()
    ear_threshold = float(row[2]) if has_ear and row[2] is not None else None

    if len(emb) != 512:
        return nom, None, ear_threshold
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb /= norm
    return nom, emb, ear_threshold


# GOOGLE SHEETS

HEADERS = ["first_name", "last_name", "uid", "timestamp", "event", "status"]

def log_to_sheets(first, last, card_id, event, status):
    def task():
        try:
            creds  = Credentials.from_service_account_file(
                JSON_KEYFILE,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            client = gspread.authorize(creds)
            sheet  = client.open_by_key(SHEET_ID).sheet1
            if sheet.row_values(1) != HEADERS:
                sheet.insert_row(HEADERS, index=1)
            sheet.append_row(
                [first, last, card_id,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 event, status],
                value_input_option="USER_ENTERED"
            )
        except Exception as e:
            print(f"[Sheets] Erreur : {e}")
    threading.Thread(target=task, daemon=True).start()


#  ENTRY / EXIT
def get_event(card_id, authorized):
    if not authorized:
        return "ENTRY"
    prev = last_event.get(card_id)
    if prev == "ENTRY":
        last_event[card_id] = "EXIT"
        return "EXIT"
    else:
        last_event[card_id] = "ENTRY"
        return "ENTRY"


#  MOBILEFACENET

session  = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

def get_embedding_mfn(face_bgr_112):
    rgb  = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
    f    = (rgb.astype(np.float32) - 127.5) / 128.0
    inp  = np.expand_dims(np.transpose(f, (2, 0, 1)), 0)
    emb  = session.run([out_name], {inp_name: inp})[0][0]
    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 1e-6 else emb.astype(np.float32)


#  gÃ©omÃ©trie

_model_points = np.array([
    (0.0, 0.0, 0.0),         (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),  (225.0, 170.0, -135.0),
    (-150.0,-150.0,-125.0),   (150.0,-150.0,-125.0)
], dtype="double")
_dst_pts = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014],
    [56.0252, 71.7366], [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def frames_alignements_pts(lm, w, h):
    le_x = (lm[33].x + lm[133].x) / 2.0 * w
    le_y = (lm[33].y + lm[133].y) / 2.0 * h
    re_x = (lm[362].x + lm[263].x) / 2.0 * w
    re_y = (lm[362].y + lm[263].y) / 2.0 * h
    return np.array([
        [le_x, le_y], [re_x, re_y],
        [lm[1].x*w,   lm[1].y*h],
        [lm[61].x*w,  lm[61].y*h],
        [lm[291].x*w, lm[291].y*h]
    ], dtype=np.float32)

def robust_points_fast(pts_list, trim=0.2):
    a = np.asarray(pts_list, dtype=np.float32)
    if a.size == 0: return None, None
    if a.ndim == 2 and a.shape == (5, 2): a = a[np.newaxis]
    N_ = a.shape[0]; k = int(np.floor(N_ * trim))
    srt = np.sort(a, axis=0)
    t   = srt if 2*k >= N_ else srt[k:N_-k]
    return np.mean(t, axis=0).astype(np.float32), float(np.mean(np.std(t, axis=0)))

def align_face(sq, pts):
    M, _ = cv2.estimateAffinePartial2D(
        pts, _dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return cv2.warpAffine(sq, M, (112, 112), borderValue=0.0)

def preprocess(img):
    ycrcb     = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    img       = cv2.cvtColor(cv2.merge([clahe.apply(y), cr, cb]), cv2.COLOR_YCrCb2BGR)
    img       = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    flou      = cv2.GaussianBlur(img, (3, 3), 1)
    return np.clip(img.astype(np.float32)
                   + 1.5*(img.astype(np.float32) - flou.astype(np.float32)),
                   0, 255).astype(np.uint8)

def is_aligned_verif(lm, w=480, h=480):
    idx = [4, 152, 33, 263, 61, 291]
    pts = np.array([(lm[i].x*w, lm[i].y*h) for i in idx], dtype="double")
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
    ok, rvec, tvec = cv2.solvePnP(_model_points, pts, cam, np.zeros((4, 1)),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return False
    rmat, _ = cv2.Rodrigues(rvec)
    euler   = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))[6]
    return abs(float(euler.flatten()[1])) <= 15 and abs(float(euler.flatten()[0]) - 180) <= 15

#  anti-spoofing 

def calcul_EAR(pts_g, pts_d):
    def ear_one(p):
        A = np.linalg.norm(p[4]-p[12]); B = np.linalg.norm(p[5]-p[11])
        C = np.linalg.norm(p[3]-p[13]); D = np.linalg.norm(p[6]-p[10])
        E = np.linalg.norm(p[2]-p[14]); F = np.linalg.norm(p[0]-p[3])
        return (A+B+C+D+E) / (5.0*F) if F > 1e-6 else 0.0
    g = ear_one(pts_g); d = ear_one(pts_d)
    return g, d, (g+d)/2.0

def make_camera_matrix(w, h):
    return np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)

def get_head_pose(landmarks, cam_mat, w, h):
    pts = np.array([(landmarks[i].x*w, landmarks[i].y*h)
                    for i in POSE_INDICES], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, pts, cam_mat,
                                   DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    euler   = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))[6].flatten()
    return float(euler[1]), float(euler[0]) - 180.0

class BlinkDetector:
    def __init__(self):
        self.last_blink_time = None
        self._blinking       = False

    def update(self, ear, threshold, now):
        if not self._blinking:
            if ear <= threshold: self._blinking = True
        elif ear > threshold:
            self._blinking       = False
            self.last_blink_time = now

    def dt(self, now):
        return None if self.last_blink_time is None else now - self.last_blink_time


#  VÃ©rification unifiÃ©

def run_verification(ear_threshold: float, ref_emb: np.ndarray,
                     full_name: str) -> str:
    if ear_threshold is None:
        print("   Seuil EAR absent : vÃ©rification refusÃ©e.")
        return "FRAUD"

    cap = cv2.VideoCapture(PC_STREAM_URL)
    if not cap.isOpened():
        print("    Stream camÃ©ra inaccessible.")
        return "FRAUD"

    
    latest_result = [None]
    lock          = threading.Lock()

    def on_result(result, _img, _ts):
        with lock:
            latest_result[0] = result

    opts_live = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARK_MODEL),
        output_face_blendshapes=False, num_faces=1,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=on_result,
    )

    
    opts_img      = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=LANDMARK_MODEL),
        running_mode=vision.RunningMode.IMAGE)
    landmarker_img = vision.FaceLandmarker.create_from_options(opts_img)

    
    phase          = "liveness"   
    cam_mat        = None
    t0_ms          = int(time.perf_counter() * 1000)
    start          = time.perf_counter()

    
    smooth_buf     = deque(maxlen=EAR_SMOOTH_FRAMES)
    detector_blink = BlinkDetector()
    no_ear         = 0
    candidate      = None; candidate_n = 0

    
    aligned_pts    = []
    final_status   = "DENIED"
    score          = 0.0

    with vision.FaceLandmarker.create_from_options(opts_live) as lmk_live:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05); continue

            frame  = cv2.flip(frame, 1)
            now    = time.perf_counter()
            hf, wf = frame.shape[:2]
            dim    = min(hf, wf)
            sq     = cv2.resize(
                frame[(hf-dim)//2:(hf-dim)//2+dim,
                      (wf-dim)//2:(wf-dim)//2+dim], (480, 480))

            if cam_mat is None:
                cam_mat = make_camera_matrix(480, 480)

            
            #  Phase 1 : EAR
            
            if phase == "liveness":
                if now - start > LIVENESS_TIMEOUT:
                    cap.release(); landmarker_img.close()
                    cv2.destroyAllWindows()
                    return "FRAUD"

                ts = int(now * 1000) - t0_ms
                lmk_live.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)), ts)

                with lock:
                    result = latest_result[0]

                if result and result.face_landmarks:
                    face       = result.face_landmarks[0]
                    yaw, pitch = get_head_pose(face, cam_mat, 480, 480)
                    aligned    = (yaw is not None
                                  and abs(yaw) <= YAW_MAX
                                  and abs(pitch) <= PITCH_MAX)

                    xs    = [lm.x for lm in face]
                    ys    = [lm.y for lm in face]
                    color = (0, 255, 0) if aligned else (0, 0, 255)
                    cv2.rectangle(sq,
                        (int(min(xs)*480), int(min(ys)*480)),
                        (int(max(xs)*480), int(max(ys)*480)), color, 2)

                    if aligned:
                        no_ear = 0
                        pts_g  = np.array([(face[i].x*480, face[i].y*480)
                                           for i in INDICES_G])
                        pts_d  = np.array([(face[i].x*480, face[i].y*480)
                                           for i in INDICES_D])
                        _, _, ear = calcul_EAR(pts_g, pts_d)
                        smooth_buf.append(ear)
                        detector_blink.update(float(np.mean(smooth_buf)),
                                              ear_threshold, now)
                        cv2.putText(sq, f"Clignez des yeux ({ear_threshold:.3f})",
                                    (10, 30), FONT_V, 0.65, color, 2)
                    else:
                        no_ear += 1
                        cv2.putText(sq, "Alignez votre visage",
                                    (10, 30), FONT_V, 0.65, color, 2)
                else:
                    no_ear += 1
                    cv2.putText(sq, "Aucun visage...",
                                (10, 30), FONT_V, 0.65, (0, 0, 255), 2)

                
                dt = detector_blink.dt(now)
                if dt is None:
                    raw = None
                elif no_ear >= NO_EAR_MAX_FRAMES:
                    raw = ("Not a real person", (0, 0, 255))
                elif dt <= MAX_BLINK_INTERVAL:
                    raw = ("Real person", (0, 255, 0))
                else:
                    raw = ("Not a real person", (0, 0, 255))

                if raw is None:
                    candidate = None; candidate_n = 0
                elif raw == candidate:
                    candidate_n += 1
                    if candidate_n >= LABEL_STABLE_N:
                        if raw[0] == "Real person":
                            phase       = "recognition"
                            aligned_pts = []
                        else:
                            cv2.putText(sq, "Not a real person",
                                        (80, 240), FONT_V, 1.4, (0, 0, 255), 3)
                            cv2.imshow("Verification", sq)
                            cv2.waitKey(1500)
                            cap.release(); landmarker_img.close()
                            cv2.destroyAllWindows()
                            return "FRAUD"
                else:
                    candidate, candidate_n = raw, 1

            
            #  Phase 2 : reconaissance de visage
            
            elif phase == "recognition":
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
                res    = landmarker_img.detect(mp_img)

                if res.face_landmarks:
                    lm      = res.face_landmarks[0]
                    aligned = is_aligned_verif(lm)
                    color   = (0, 255, 0) if aligned else (0, 0, 255)
                    xs = [l.x for l in lm]; ys = [l.y for l in lm]
                    cv2.rectangle(sq,
                        (int(min(xs)*480), int(min(ys)*480)),
                        (int(max(xs)*480), int(max(ys)*480)), color, 2)
                    cv2.putText(sq,
                        "Alignez votre visage" if not aligned else full_name,
                        (10, 30), FONT_V, 0.65, color, 2)

                    if aligned:
                        aligned_pts.append(frames_alignements_pts(lm, 480, 480))
                        if len(aligned_pts) >= N_FRAMES:
                            pts_f, stab = robust_points_fast(aligned_pts, 0.2)
                            if pts_f is not None and stab < STAB_SEUIL:
                                face  = align_face(sq, pts_f).astype(np.uint8)
                                face  = preprocess(face)
                                emb   = get_embedding_mfn(face)
                                score = float(np.dot(emb, ref_emb))
                                final_status = ("AUTHORIZED"
                                                if score >= THRESHOLD else "DENIED")
                                phase = "done"
                            aligned_pts.clear()
                else:
                    cv2.putText(sq, "Aucun visage...",
                                (10, 30), FONT_V, 0.65, (0, 0, 255), 2)

            
            #  Phase 3 : RÃ‰SULTAT FINAL
            
            elif phase == "done":
                color_res = (0, 255, 0) if final_status == "AUTHORIZED" else (0, 0, 255)
                cv2.putText(sq, final_status,
                            (80, 240),  FONT_V, 1.4, color_res, 3)
                cv2.putText(sq, f"score: {score:.3f}",
                            (150, 290), FONT_V, 0.6, color_res, 1)
                cv2.imshow("Verification", sq)
                cv2.waitKey(int(RESULT_DURATION * 1000))
                cap.release(); landmarker_img.close()
                cv2.destroyAllWindows()
                return final_status

            
            cv2.imshow("Verification", sq)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release(); landmarker_img.close()
                cv2.destroyAllWindows()
                return "FRAUD"


#  Point D'entrÃ©e

if __name__ == "__main__":
    print("\n=== SYSTÃˆME PRÃŠT â€” SCANNEZ VOTRE BADGE ===\n")

    while True:
        try:
            card_id = input("RFID ID > ").strip()
        except KeyboardInterrupt:
            print("\nArrÃªt.")
            break
        if not card_id:
            continue

        nom, ref_emb, ear_threshold = db_get_person(card_id)

        if ref_emb is None:
            reason = "Embedding invalide" if nom else f"Badge inconnu : {card_id}"
            print(f"  {reason}")
            parts = (nom or "Inconnu").split(maxsplit=1)
            log_to_sheets(parts[0], parts[1] if len(parts) > 1 else "-",
                          card_id, "ENTRY", "DENIED")
            continue

        
        result = run_verification(ear_threshold, ref_emb, nom)

        if result == "FRAUD":
            print(f"  FRAUDE DÃ‰TECTÃ‰E => {nom} ({card_id})  annulÃ©.")
            continue

        status = result   # "AUTHORIZED" ou "DENIED"
        event  = get_event(card_id, status == "AUTHORIZED")
        print(f"=> {nom}  |  {event}  |  {status}")

        parts = nom.split(maxsplit=1)
        log_to_sheets(parts[0], parts[1] if len(parts) > 1 else "-",
                      card_id, event, status)