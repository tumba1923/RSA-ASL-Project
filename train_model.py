from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocess_data import X_train, X_test, y_train, y_test, GESTURES

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

model.save("asl_lstm_model.h5")
print("Model saved as 'asl_lstm_model.h5'")
