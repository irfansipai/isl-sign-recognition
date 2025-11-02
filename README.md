-----

## Real-Time Indian Sign Language Recognition (0-9, A-Z) with Speech Output

This project captures video from a webcam, detects hand landmarks in real-time using MediaPipe, and predicts the corresponding Indian Sign Language (ISL) sign (digits 0-9 and letters A-Z) using a pre-trained TensorFlow Keras model (`model.h5`).

**Upgrade:** This version builds upon an initial gesture detection project by adding asynchronous text-to-speech output and prediction debouncing to significantly improve usability and performance.

-----
![Demo](assets/demo.gif)

### How it Works (Pipeline)

1.  **Capture:** A dedicated thread (`WebcamStream`) continuously reads frames from the webcam using OpenCV.
2.  **Detect & Preprocess (Conditional):** On selected frames (to manage performance):
      * Detect hand landmarks (up to two hands) using MediaPipe Hands.
      * If hands are found, calculate landmark coordinates relative to the image.
      * Preprocess landmarks: Normalize coordinates by centering them relative to the wrist and scaling them based on hand size. This creates a consistent input for the model.
3.  **Predict:** Feed the normalized landmarks (as a Pandas DataFrame) into the pre-trained Keras model (`model.h5`) to get a prediction (0-9, A-Z).
4.  **Debounce:** Store the prediction in a history buffer (`deque`). Only consider the prediction "stable" if it remains the same for a defined number of consecutive frames (`STABILITY_THRESHOLD`).
5.  **Speak (Async):** If a prediction is stable *and* different from the last spoken sign:
      * Clear any pending speech tasks from a dedicated queue (`queue.Queue`).
      * Add the new stable sign to the speech queue.
      * A separate worker thread reads from the queue and uses `pyttsx3` to speak the sign aloud without blocking the main video loop.
6.  **Visualize:** Use OpenCV to draw the detected hand landmarks, connections, and the *current* (potentially unstable) predicted sign onto the live video frame.
7.  **Display:** Show the annotated video frame using `cv2.imshow()`.

-----

### Key Features

  * **Real-Time Hand Detection:** Utilizes Google's MediaPipe Hands for robust multi-hand landmark detection.
  * **Landmark Preprocessing:** Implements normalization techniques (centering, scaling) for consistent model input.
  * **ML Model Inference:** Integrates a TensorFlow Keras model (`model.h5`) trained on landmark data derived from an ISL dataset.
  * **Visual Feedback:** Overlays detected landmarks, connections, and the current prediction onto the video stream using OpenCV.
  * **Prediction Debouncing:** Uses a `deque` buffer to ensure prediction stability over several frames, preventing flickering output.
  * **Asynchronous Text-to-Speech:** Leverages Python's `threading` and `queue` with `pyttsx3` for non-blocking audio feedback.
  * **Basic Frame Skipping:** Processes only a subset of frames to manage computational load.

-----

### Technologies Used

  * **Programming Language:** Python
  * **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mediapipe`)
  * **Machine Learning:** TensorFlow (`keras`), NumPy, Pandas
  * **Concurrency:** `threading`, `queue`
  * **Text-to-Speech:** `pyttsx3`

-----

### Challenges & Learning Points

  * **Real-Time Performance:** Balancing landmark detection, model inference, and rendering. Frame skipping was explored as a basic optimization.
  * **Prediction Stability:** Addressed flickering predictions by implementing a debouncing mechanism using `deque`.
  * **Non-Blocking Output:** Solved the issue of speech output blocking the video loop by implementing an asynchronous, queue-based speech worker thread.
  * **Model Integration:** Successfully loaded and performed inference with a Keras `.h5` model within the real-time loop.
  * **Accuracy Limitation:** Acknowledged that the accuracy of the underlying `model.h5` could be further improved with more data or a different architecture.

-----

### Examples

-----

### File Descriptions (From Original Project)

  * `isl_detection.py`: Main script for real-time detection (later upgraded).
  * `dataset_keypoint_generation.py`: Script to convert image dataset to landmarks.
  * `keypoint.csv`: Landmark data generated from the dataset.
  * `ISL_classifier.ipynb`: Jupyter Notebook used for training the classifier.
  * `model.h5`: The trained TensorFlow Keras classifier model.

-----

## Setup and Installation

Follow these steps to set up and run the Indian Sign Language Recognition project on your local machine.

### Prerequisites

  * **Git:** You need Git installed to clone the repository.
  * **Python:** The required Python version depends on your operating system:
      * **Windows:** Python **3.9.13** is recommended. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3913/) or install it via the Microsoft Store.
      * **Linux:** Python **3.9.18** is recommended. You can install it using your system's package manager (like `apt` on Debian/Ubuntu) or manage multiple Python versions using a tool like [`pyenv`](https://www.google.com/search?q=%5Bhttps://github.com/pyenv/pyenv%5D\(https://github.com/pyenv/pyenv\)).
  * **Webcam:** A connected webcam is required for real-time detection.

### Installation Steps

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:

    ```bash
    git clone <your-repository-url>
    cd isl-sign-recognition # Or your chosen repository name
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies. Navigate into the project directory and run:

    ```bash
    python -m venv venv
    ```

    *(If you have multiple Python versions, ensure you use the correct one, e.g., `python3.9 -m venv venv`)*

3.  **Activate the Virtual Environment:**

      * **On Windows (Command Prompt/PowerShell):**
        ```bash
        venv\Scripts\activate
        ```
      * **On Linux/macOS (Bash/Zsh):**
        ```bash
        source venv/bin/activate
        ```

    After activation, you should see `(venv)` at the beginning of your terminal prompt. You can verify the Python version by running `python --version`.

4.  **Install Dependencies:**
    Install the required packages using the correct requirements file for your OS:

      * **On Windows:**
        ```bash
        pip install -r requirements-windows.txt
        ```
      * **On Linux:**
        ```bash
        pip install -r requirements-linux.txt
        ```

### Running the Application

1.  **Ensure your virtual environment is activated.** (You should see `(venv)` in your prompt).
2.  **Run the main script:**
    ```bash
    python isl_detection.py
    ```
3.  The application should start, open your webcam feed in a window, and begin detecting hand signs.
4.  **To stop the application:** Press the **`Esc`** key while the OpenCV window is active.

-----