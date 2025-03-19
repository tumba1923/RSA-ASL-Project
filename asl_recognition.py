import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("asl_lstm_model.h5")

# Define ASL gestures (same order as during training)
GESTURES = ["Hello", "Thank You", "Yes", "No", "I Love You", "Help", "Please", "Sorry", "Stop"]
SEQUENCE_LENGTH = 30  # Frames per gesture

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Store frames for prediction
sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y, z) coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sequence.append(landmarks)

            # Maintain only the last 30 frames
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # Make a prediction if we have enough frames
            if len(sequence) == SEQUENCE_LENGTH:
                prediction = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_label = np.argmax(prediction)
                confidence = prediction[predicted_label]

                # Display the recognized ASL gesture
                if confidence > 0.7:  # Confidence threshold
                    predicted_word = GESTURES[predicted_label]
                    cv2.putText(frame, f"{predicted_word} ({confidence:.2f})", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
