import cv2
import tkinter as tk
from tkinter import Label, Button, StringVar
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
import pyttsx3  # Import the pyttsx3 library

# Custom DepthwiseConv2D to ignore the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# Initialize model and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C", "H", "I", "K", "L", "S", "T"]  # Labels from your model

# Variables for word detection
current_word = ""
last_pred = None
last_time = time.time()
delay_between_letters = 1  # Time in seconds required to confirm a letter
delay_for_space = 1  # Time in seconds to add a space when no hand is detected

# Initialize voice engine
engine = pyttsx3.init()

# Initialize Tkinter window
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("1000x700")
root.configure(bg="#2E2E2E")  # Dark background for the window

# Label to display recognized text
recognized_text = StringVar()
text_label = Label(root, textvariable=recognized_text, font=("Helvetica", 36, "bold"), bg="#2E2E2E", fg="#FFFFFF")
text_label.pack(pady=20, fill="both")

# Canvas to display video feed
video_frame = Label(root, bg="#2E2E2E")
video_frame.pack(pady=10)

# List to store hand positions for movement tracking
hand_positions = []

# Function to update video feed and perform detection
def update_frame():
    global last_pred, last_time, current_word, hand_positions

    success, img = cap.read()
    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        img_width = imgOutput.shape[1]

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Track hand's center position
            center_x = x + w // 2
            center_y = y + h // 2

            # Store the position of the hand's center
            hand_positions.append((center_x, center_y))

            # Keep the list size small to only track the last few frames
            if len(hand_positions) > 20:  # Store the last 20 frames
                hand_positions.pop(0)

            # Draw the trajectory of the hand
            for i in range(1, len(hand_positions)):
                # Line thickness depends on speed of movement (difference between points)
                thickness = int(math.sqrt(i) * 2)
                cv2.line(imgOutput, hand_positions[i - 1], hand_positions[i], (0, 255, 0), thickness)

            # Your existing logic for hand detection and prediction
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Your existing logic for word detection
            if last_pred == labels[index]:
                if time.time() - last_time > delay_between_letters:
                    current_word += labels[index]
                    last_pred = None  # Reset last_pred to start detecting the next letter
            else:
                last_pred = labels[index]
                last_time = time.time()

            # Draw the predicted letter
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

        else:
            # If no hands are detected, add a space after the delay
            if time.time() - last_time > delay_for_space:
                current_word += " "
                last_pred = None
                last_time = time.time()

        # Update the recognized text
        recognized_text.set(current_word)

        # Convert image to RGB and display it in the Tkinter window
        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(imgOutput)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)

    root.after(10, update_frame)

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start button to begin detection
def start_detection():
    speak("Program has started")
    update_frame()

# Stop button to end detection
def stop_detection():
    speak("Program has stopped")
    cap.release()
    root.quit()

# Reset button to clear the recognized text
def reset_detection():
    global current_word
    current_word = ""
    recognized_text.set("")
    speak("Text has been reset")

# Add buttons for start, stop, and reset
button_frame = tk.Frame(root, bg="#2E2E2E")
button_frame.pack(pady=20)

start_button = Button(button_frame, text="Start", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="#FFFFFF", command=start_detection)
start_button.grid(row=0, column=0, padx=10, pady=5)

reset_button = Button(button_frame, text="Reset", font=("Helvetica", 18, "bold"), bg="#FFC107", fg="#FFFFFF", command=reset_detection)
reset_button.grid(row=0, column=1, padx=10, pady=5)

stop_button = Button(button_frame, text="Stop", font=("Helvetica", 18, "bold"), bg="#F44336", fg="#FFFFFF", command=stop_detection)
stop_button.grid(row=0, column=2, padx=10, pady=5)

# Run the Tkinter event loop
root.mainloop()
