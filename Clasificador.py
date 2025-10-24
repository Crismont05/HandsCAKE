import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- CONFIGURACIÓN DEL CLASIFICADOR ---
try:
    with open('clasificador_letras.pkl', 'rb') as file:
        data = pickle.load(file)
        classifier = data['model']
except FileNotFoundError:
    print("¡Error! No se encontró 'clasificador_letras.pkl'. Ejecuta el código de entrenamiento (Parte 2) primero.")
    exit()

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Función de normalización 
def extract_and_normalize_landmarks(hand_landmarks):
    """Extrae las 63 coordenadas (x, y, z) y las normaliza respecto a la muñeca (landmark 0)."""
    
    landmarks = []
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z
    
    for landmark in hand_landmarks.landmark:
        landmarks.extend([
            (landmark.x - wrist_x) * 1000,
            (landmark.y - wrist_y) * 1000,
            (landmark.z - wrist_z) * 1000
        ])
        
    return np.array(landmarks).reshape(1, -1) # Devolver como un array 1x63

# --- CAPTURA DE VIDEO Y BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)
predicted_letter = "Esperando..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Voltear para efecto espejo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Dibujar los puntos de referencia
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 2. Extraer y Normalizar las coordenadas
            try:
                hand_features = extract_and_normalize_landmarks(hand_landmarks)
                
                # 3. Clasificar (Predecir la letra)
                prediction = classifier.predict(hand_features)
                predicted_letter = prediction[0]
                
            except Exception as e:
                predicted_letter = "Error"

    # 4. Mostrar la predicción en el frame
    cv2.rectangle(frame, (0, 0), (300, 50), (100, 100, 100), -1) # Fondo gris
    cv2.putText(frame, f"Prediccion: {predicted_letter}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Clasificador de Letras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()