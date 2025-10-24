import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# --- Configuración ---
DATA_DIR = 'C:\ImportantProyects\HandsCAKE\DataSetSignos' # ruta donde esta el dataset
CSV_FILE = 'datos_entrenamiento.csv'
hands_model = mp.solutions.hands
hands = hands_model.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Lista para almacenar las características (coordenadas) y las etiquetas
data = []

# Función de normalización de puntos de referencia
def extract_and_normalize_landmarks(hand_landmarks):
    """Extrae las 63 coordenadas (x, y, z) y las normaliza respecto a la muñeca (landmark 0)."""
    
    landmarks = []
    # Usamos el punto 0 (muñeca) como punto de referencia
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    for landmark in hand_landmarks.landmark:
        # Normalizar y aplanar
        landmarks.extend([
            (landmark.x - wrist_x) * 1000,
            (landmark.y - wrist_y) * 1000,
            (landmark.z - wrist_z) * 1000
        ])
        
    return landmarks

# --- Proceso de Extracción ---
print("Iniciando la extracción de coordenadas...")
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if os.path.isdir(label_path):
        print(f"Procesando letra: {label}")
        
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            
            # Cargar y convertir la imagen
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen para detección de manos
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Solo tomamos la primera mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extraer características
                features = extract_and_normalize_landmarks(hand_landmarks)
                
                # Añadir la etiqueta
                features.append(label)
                data.append(features)

print("Extracción completada. Guardando CSV...")

# Crear y guardar el DataFrame
df = pd.DataFrame(data)
df.to_csv(CSV_FILE, index=False, header=False) # No headers, solo datos
print(f"Datos guardados en {CSV_FILE}. ¡Listo para entrenar!")