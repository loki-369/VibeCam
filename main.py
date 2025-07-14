import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# Load pre-trained emotion model
model = load_model('emotion_model.h5', compile=False)

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotions and matching emojis
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emojis = ['😠', '🤢', '😨', '😄', '😢', '😲', '😐']

# Load emoji-capable font (Segoe UI Emoji for Windows)
emoji_font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 24)

# Log file setup
log_file = open("emotion_log.csv", "a", encoding="utf-8")
log_file.write("Timestamp,Emotion,Confidence\n")

def draw_text_with_emoji(frame, text, position):
    # Convert OpenCV BGR image to PIL RGB image
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)
    draw.text(position, text, font=emoji_font, fill=(255, 255, 255))

    # Convert back to OpenCV BGR
    frame[:] = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))         # Ensure input shape matches model
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = model.predict(face, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = emotions[emotion_index]
        emoji = emojis[emotion_index]
        confidence = prediction[emotion_index] * 100

        # Draw box and emotion
        text = f"{emoji} {emotion} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        draw_text_with_emoji(frame, text, (x, y - 30))

        # Log to file
        log_file.write(f"{datetime.now()},{emotion},{confidence:.2f}\n")

    # Show output
    cv2.imshow('VibeCam 😎', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Cleanup
log_file.close()
cap.release()
cv2.destroyAllWindows()
