import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import threading

# Load model and face detector
model = load_model('emotion_model.h5', compile=False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotions and emojis
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']
emoji_font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 24)

# Log file
log_file = open("emotion_log.csv", "a", encoding="utf-8")
log_file.write("Timestamp,Emotion,Confidence\n")

# Global variable for graph
latest_probs = np.zeros(len(emotions))

# Graph update in a separate thread
def start_graph():
    plt.ion()
    fig, ax = plt.subplots()
    bars = ax.bar(emotions, latest_probs, color='skyblue')
    ax.set_ylim([0, 100])
    ax.set_title('Live Emotion Confidence')
    ax.set_ylabel('% Confidence')

    while True:
        for bar, new_val in zip(bars, latest_probs):
            bar.set_height(new_val)
        fig.canvas.draw()
        fig.canvas.flush_events()

# Start graph thread
graph_thread = threading.Thread(target=start_graph, daemon=True)
graph_thread.start()

def draw_text_with_emoji(frame, text, position):
    # Convert to PIL image
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=emoji_font, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# Webcam loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = model.predict(face, verbose=0)[0]
        latest_probs[:] = prediction * 100  # update global variable for graph

        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]
        emoji = emojis[emotion_index]
        confidence = prediction[emotion_index] * 100

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emoji} {emotion} ({confidence:.1f}%)"
        draw_text_with_emoji(frame, text, (x, y - 30))

        # Log to file
        log_file.write(f"{datetime.now()},{emotion},{confidence:.2f}\n")

    cv2.imshow('VibeCam 😎', frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
