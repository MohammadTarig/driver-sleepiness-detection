# **Driver Sleepiness Detection App 😴🚗**

A real-time desktop app to detect driver sleepiness and yawning using facial landmarks, powered by OpenCV, Dlib, and Kivy.

---

## 💡 Features

- 🔍 Live detection of drowsiness using Eye Aspect Ratio (EAR)
- 😮 Yawning detection using Mouth Aspect Ratio (MAR)
- 🔔 Audio alert system when drowsiness or yawning is detected
- 🎛️ GUI interface with Pause/Resume functionality (built with Kivy)

---

## 🛠 Installation Guide (Anaconda Setup Recommended)

1. **Download and Unzip**:
   - Download the Driver Sleepiness Detection App zipped folder and unzip it.

2. **Install Anaconda**:
   - Download and install Anaconda from [www.anaconda.com/download](https://www.anaconda.com/download).

3. **Open Anaconda Prompt**:
   - Launch Anaconda Prompt from your start menu.

4. **Create a Virtual Environment**:
     ```sh
     conda create -n dsdenv python=3.12
     ```

5. **Activate the Virtual Environment**:
     ```sh
     conda activate dsdenv
     ```

6. **Install Required Packages**:
   - Install the following packages in the virtual environment:

     - **dlib**:
       ```sh
       conda install -c conda-forge dlib
       ```
     - **OpenCV**:
       ```sh
       conda install -c conda-forge opencv
       ```
     - **Kivy**:
       ```sh
       conda install -c conda-forge kivy
       # or if issues:
       pip install kivy

       ```
     - **numpy**:
       ```sh
       conda install numpy
       ```
     - **scipy**:
       ```sh
       conda install scipy
       ```
     - **playsound**:
       ```sh
       pip install playsound
       ```
     - **imutils**:
       ```sh
       pip install imutils
       ```
7. **Download the facial landmark model**:
   - Download the facial landmark model here: 🔗 [shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models)
   - Put this file inside the ```shape_predictor``` folder

---

## 📂 Setup Files
**Ensure the following are in the correct locations:**
```
driver-sleepiness-detection/
├── audio/
│   ├── audio_file1.wav
│   ├── audio_file2.wav
│   ├── ...
├── shape_predictor/
│   └── shape_predictor_68_face_landmarks.dat
├── main.py
└── README.md

```

---

## ▶️ How to Run the Application

1. **Navigate to the Application Directory**:
   - In Anaconda Prompt, navigate to the directory where you downloaded the Driver Sleepiness Detection App folder:
     ```sh
     cd <Driver_Sleepiness_Detection_App_directory>
     ```

2. **Run the Application**:
   - Execute the application by running:
     ```sh
     python main.py
     ```

✅ You're all set! The Driver Sleepiness Detection App should now be up and running.

## 📌 Notes
- Best results in good lighting
- Requires webcam access
- Detection may vary depending on face visibility and camera angle