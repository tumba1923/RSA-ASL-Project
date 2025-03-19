import cv2
import numpy as np
import mediapipe as mp
import os

GESTURES = ["Hello", "Thank You", "Yes", "No", "I Love You", "Help", "Please", "Sorry", "Stop"]
DATA_PATH = "asl_data"
SEQUENCE_LENGTH = 30
SAMPLES_PER_CLASS = 100

for gesture in GESTURES:
    os.makedirs(os.path.join(DATA_PATH, gesture), exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f"Collecting data for: {gesture}")
    for sample in range(SAMPLES_PER_CLASS):
        frames = []
        print(f"Sample {sample+1}/{SAMPLES_PER_CLASS}")

        while len(frames) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                                [lm.y for lm in hand_landmarks.landmark] + \
                                [lm.z for lm in hand_landmarks.landmark]
                    frames.append(landmarks)

            cv2.putText(frame, f"Recording: {gesture} ({sample+1}/{SAMPLES_PER_CLASS})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("ASL Data Collection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(frames) == SEQUENCE_LENGTH:
            np.save(os.path.join(DATA_PATH, gesture, f"{sample}.npy"), np.array(frames))

cap.release()
cv2.destroyAllWindows()
