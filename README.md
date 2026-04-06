# Système de Contrôle d'Accès par Reconnaissance Faciale et Badge RFID

**Projet de Fin d'Études — DUT en Système Embarqués industriel (SEI)**  
**École Supérieure de Technologie de Salé (ESTS)**  
**Université Mohammed V — Rabat**

---

## Résumé

Ce projet propose un système embarqué de contrôle d'accès combinant la lecture de badge RFID et la reconnaissance faciale en temps réel. Il intègre une détection anti-spoofing par analyse du clignement des yeux (Eye Aspect Ratio), une base de données locale synchronisée entre un PC d'enrôlement et un Raspberry Pi, ainsi qu'une journalisation automatique des accès vers Google Sheets.

---

## Architecture Générale

```
┌─────────────────────────────────────────────────────┐
│                   PC d'Enrôlement                   │
│  database_setup.py — Capture, alignement, embedding │
│  server_cam.py     — Serveur flux vidéo (MJPEG)     │
└────────────────────────┬────────────────────────────┘
                         │  HTTP (LAN) - Ethernet Rj45
                         ▼
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi                       │
│  get_file.py      — Réception de la base faces.db   │
│  verification.py  — Vérification RFID + visage      │
└────────────────────────┬────────────────────────────┘
                         │  API Google Sheets
                         ▼
                  Journal des accès (cloud) --- > Google Looker (Dashboard)
```

---

## Fonctionnalités

- **Enrôlement** : capture de 10 frames alignées par stabilité géométrique, calcul d'un embedding facial moyen (512 dimensions) via MobileFaceNet.
- **Anti-spoofing passif** : calibration automatique du seuil EAR (Eye Aspect Ratio) à l'enrôlement, puis détection de clignement réel à la vérification.
- **Pose de tête** : rejet des visages non frontaux via estimation PnP (yaw ≤ 15°, pitch ≤ 15°).
- **Reconnaissance** : comparaison cosinus entre l'embedding en direct et la référence stockée (seuil = 0.5).
- **Synchronisation** : envoi automatique de `faces.db` au Raspberry Pi après chaque enrôlement.
- **Journalisation** : chaque événement (ENTRÉE / SORTIE / REFUS / FRAUDE) est enregistré dans Google Sheets avec horodatage.

---

## Structure du Dépôt

```
.
├── database_setup.py   # Enrôlement des personnes (PC)
├── verification.py     # Vérification RFID + visage (Raspberry Pi)
├── get_file.py         # Serveur Flask de réception de la DB (Raspberry Pi)
├── server_cam.py       # Serveur de flux vidéo MJPEG (PC)
├── faces.db            # Base SQLite (générée à l'exécution)
├── w600k_mbf.onnx      # Modèle MobileFaceNet (non inclus — voir ci-dessous)
└── face_landmarker.task # Modèle MediaPipe (non inclus — voir ci-dessous)
```

---

## Prérequis

### Matériel
- PC (Windows/Linux) avec webcam
- Raspberry Pi (3B+ ou supérieur) connecté au même réseau local
- Lecteur RFID (USB ou UART, connecté au Raspberry Pi)

### Logiciel

```bash
pip install opencv-python mediapipe onnxruntime numpy flask requests gspread google-auth
```

### Modèles à télécharger manuellement

| Fichier | Source |
|---|---|
| `w600k_mbf.onnx` | [Face reidentification](https://github.com/yakhyo/face-reidentification/releases) |
| `face_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) |

### Credentials Google Sheets
Créer un compte de service Google Cloud, télécharger le fichier JSON et le placer à la racine du projet. Renseigner son chemin dans `JSON_KEYFILE` et l'identifiant de la feuille dans `SHEET_ID` (dans `verification.py`).

---

## Configuration Réseau

| Variable | Fichier | Valeur par défaut | Description |
|---|---|---|---|
| `RPI_URL` | `database_setup.py` | `http://192.168.1.11:5000/upload` | Adresse Flask du Raspberry Pi |
| `PC_STREAM_URL` | `verification.py` | `http://192.168.1.10:5000/video` | Flux caméra du PC |
| `host` | `get_file.py` | `192.168.1.11` | IP du Raspberry Pi |
| `host` | `server_cam.py` | `192.168.1.10` | IP du PC |

Adapter ces valeurs à votre configuration réseau locale.

---

## Utilisation

### 1. Sur le PC — Démarrer le serveur de flux vidéo

```bash
python server_cam.py
```

### 2. Sur le PC — Enrôler un utilisateur

```bash
python database_setup.py
```

Suivre les instructions à l'écran : saisir le prénom, le nom, l'identifiant de badge, puis présenter le visage face à la caméra et cligner des yeux pour la calibration EAR.

### 3. Sur le Raspberry Pi — Démarrer le serveur de réception de la base

```bash
python get_file.py
```

### 4. Sur le Raspberry Pi — Lancer la vérification

```bash
python verification.py
```

Scanner un badge RFID, puis présenter le visage devant la caméra. Le résultat (`AUTHORIZED`, `DENIED` ou `FRAUD`) s'affiche à l'écran et est enregistré dans Google Sheets.

---

## Base de Données

La base SQLite `faces.db` contient une table `personnes` :

| Colonne | Type | Description |
|---|---|---|
| `card_id` | TEXT (PK) | Identifiant du badge RFID |
| `full_name` | TEXT | Nom complet de la personne |
| `embedding` | BLOB | Vecteur facial 512D (float32) |
| `ear_threshold` | REAL | Seuil EAR calibré individuellement |
| `timestamp` | TEXT | Date et heure d'enrôlement |

---

## Pipeline de Vérification

```
Scan badge RFID
      │
      ▼
Recherche en DB  ──── Inconnu ──▶ DENIED (logué)
      │
      ▼
Phase 1 : Anti-spoofing (EAR)
  Clignement détecté dans 30 s ?  ──── Non ──▶ FRAUD
      │
      ▼
Phase 2 : Reconnaissance faciale
  cosine_similarity(embedding_live, embedding_ref) ≥ 0.5 ?
      │
   Oui / Non
      │
      ▼
AUTHORIZED / DENIED  ──▶  Google Sheets
```

---

## Technologies Utilisées

| Technologie | Rôle |
|---|---|
| Python 3.10 | Langage principal |
| OpenCV | Capture et traitement vidéo |
| MediaPipe Face Landmarker | Détection de 478 points faciaux |
| MobileFaceNet (ONNX) | Extraction d'embeddings faciaux |
| ONNX Runtime | Inférence du modèle de reconnaissance |
| SQLite | Stockage local des profils |
| Flask | Communication HTTP entre PC et Pi |
| gspread | Journalisation Google Sheets |

---

## Limitations et Perspectives

- Le système fonctionne en réseau local uniquement (LAN). Une extension vers une architecture cloud permettrait un déploiement multi-sites.
- La calibration EAR est individuelle et doit être refaite si les conditions d'éclairage varient fortement.
- L'intégration d'un second facteur biométrique (voix, empreinte) renforcerait la sécurité.

---

## Auteurs

Projet réalisé par Oubari Amine et Yabou Hind dans le cadre du Diplôme Universitaire de Technologie (DUT) à l'**École Supérieure de Technologie de Salé (ESTS)**, sous la supervision du corps enseignant du département 3M.

---

*Année universitaire 2024–2025*
