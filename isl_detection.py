import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
from threading import Thread
import queue
import pyttsx3
from collections import deque # Uset for efficient history buffer

# --- Threaded Webcam Class ---
class WebcamStream:
    # (No changes to this class)
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# --- Speech Worker Function ---
def speech_worker(q):
    """Processes speech tasks from a queue in a separate thread."""
    engine = pyttsx3.init()
    while True:
        try:
            text = q.get(timeout=1)
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            q.task_done()
        except queue.Empty:
            pass

# --- Model and Constants Setup ---
try:
    model = keras.models.load_model("model.h5")
except Exception as e:
    print(f"Error loading model 'model.h5': {e}")
    print("Please ensure the model file is in the correct directory.")
    exit()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# --- Landmark Processing Functions ---
# (No changes to these functions)
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if max(list(map(abs, temp_landmark_list))) != 0 else 1
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# --- Main Application Loop ---

# Start threaded video stream
vs = WebcamStream(src=0).start()

# Start speech thread
speech_queue = queue.Queue()
speech_thread = Thread(target=speech_worker, args=(speech_queue,), daemon=True)
speech_thread.start()

# --- Frame skipping variables ---
frame_counter = 0
CYCLE_LENGTH = 4
PROCESS_FRAMES = 2

# --- State persistence and debouncing variables ---
last_spoken_label = ""
STABILITY_THRESHOLD = 3  # Number of frames a prediction must be stable before speaking
prediction_history = deque(maxlen=STABILITY_THRESHOLD)
current_label = ""

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        image = vs.read()
        if image is None:
            continue

        frame_counter += 1
        # image = cv2.flip(image, 1)

        # --- Conditional Processing Block ---
        if (frame_counter % CYCLE_LENGTH) < PROCESS_FRAMES:
            debug_image = copy.deepcopy(image)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            current_label = "" # Reset label for this processing cycle

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Prediction logic
                    # 1. Convert to numpy array directly (reshape to 1 row, X columns)
                    input_data = np.array([pre_processed_landmark_list], dtype=np.float32)

                    # 2. Call model directly (skips the heavy .predict() API overhead)
                    predictions = model(input_data, training=False) 
                    
                    predicted_classes = np.argmax(predictions, axis=1)
                    current_label = alphabet[predicted_classes[0]] # Get current frame's prediction

                    # --- Drawing logic ---
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    cv2.putText(image, current_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            # --- Debouncing Logic ---
            prediction_history.append(current_label) # Add current prediction to history

        # --- Speech Trigger Logic (runs after processing check) ---
        # Check if the history buffer is full and all predictions in it are the same.
        is_stable = len(prediction_history) == STABILITY_THRESHOLD and all(p == current_label for p in prediction_history)

        if is_stable and current_label != last_spoken_label and current_label != "":
            # Clear queue of any old, unprocessed words to prevent pile-up
            while not speech_queue.empty():
                try:
                    speech_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Add new stable word to queue
            speech_queue.put(current_label)
            last_spoken_label = current_label # Update state to prevent re-speaking

        cv2.imshow('Indian sign language detector', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Cleanup
speech_queue.put(None)
speech_thread.join()
vs.stop()
cv2.destroyAllWindows()