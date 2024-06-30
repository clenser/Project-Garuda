# Real-Time Video Processing and Robotic Control using MediaPipe, OpenCV, Firebase, and Arduino

This project is a comprehensive system that integrates real-time video processing, emotion analysis, face recognition, robotic control, and cloud storage. The system uses various technologies and libraries, including MediaPipe, OpenCV, DeepFace, Firebase, PyFirmata2, and more.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

1. **Real-Time Video Processing**: Capture live video feed using OpenCV.
2. **Pose Detection**: Detect and process human poses using MediaPipe.
3. **Face Detection and Recognition**: Detect and recognize faces using DeepFace and face_recognition libraries.
4. **Emotion Analysis**: Analyze facial emotions using DeepFace.
5. **Robotic Control**: Control robotic servos based on detected poses using PyFirmata2.
6. **Firebase Integration**: Store frames and annotations in Firebase Storage.
7. **Email Notifications**: Send email notifications on user login using smtplib.
8. **Web Streaming**: Stream video feed over HTTP using Bottle.
9. **User Verification**: Implement user verification with PIN code using Tkinter.

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- DeepFace
- PyFirmata2
- smtplib
- Bottle
- Firebase Admin SDK
- face_recognition
- Tkinter
- tqdm
- requests

## Setup
# Real-Time Video Processing and Robotic Control using MediaPipe, OpenCV, Firebase, and Arduino

This project is a comprehensive system that integrates real-time video processing, emotion analysis, face recognition, robotic control, and cloud storage. The system uses various technologies and libraries, including MediaPipe, OpenCV, DeepFace, Firebase, PyFirmata2, and more.

![Project Workflow](https://github.com/your-username/your-repo/raw/main/images/project-workflow.png)

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## Features

1. **Real-Time Video Processing**: Capture live video feed using OpenCV.
2. **Pose Detection**: Detect and process human poses using MediaPipe.
3. **Face Detection and Recognition**: Detect and recognize faces using DeepFace and face_recognition libraries.
4. **Emotion Analysis**: Analyze facial emotions using DeepFace.
5. **Robotic Control**: Control robotic servos based on detected poses using PyFirmata2.
6. **Firebase Integration**: Store frames and annotations in Firebase Storage.
7. **Email Notifications**: Send email notifications on user login using smtplib.
8. **Web Streaming**: Stream video feed over HTTP using Bottle.
9. **User Verification**: Implement user verification with PIN code using Tkinter.

![System Architecture](https://github.com/your-username/your-repo/raw/main/images/system-architecture.png)

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- DeepFace
- PyFirmata2
- smtplib
- Bottle
- Firebase Admin SDK
- face_recognition
- Tkinter
- tqdm
- requests

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/real-time-video-processing-robotic-control.git
   cd real-time-video-processing-robotic-control

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/real-time-video-processing-robotic-control.git
   cd real-time-video-processing-robotic-control
   ```

2. **Install Dependencies**:

   ```bash
   pip install opencv-python mediapipe deepface pyfirmata2 smtplib bottle firebase-admin face-recognition tqdm requests
   ```

3. **Firebase Setup**:
   - Place your Firebase service account key at `C:\Users\srbar\OneDrive\Desktop\New folder\serviceAccountKey.json`.
   - Update the Firebase bucket name in the code.

4. **Configure Email**:
   - Update the email credentials in the `send_email` function.

## Usage

1. **Run the Application**:

   ```bash
   python main.py
   ```

2. **Web Streaming**:
   - Access the video feed at `http://<your-ip>:8080/video_feed`.

3. **User Verification**:
   - Follow the prompts on the Tkinter GUI for face recognition and PIN code input.

4. **Control Robot**:
   - The robotic servos will move based on the detected poses.

5. **Data Upload**:
   - Frames and annotations are stored locally and uploaded to Firebase Storage.

## Project Structure

```
real-time-video-processing-robotic-control/
│
├── main.py                             # Main application code
├── requirements.txt                    # List of dependencies
├── README.md                           # Project documentation
└── C:\Users\srbar\OneDrive\Desktop\New folder\Image_annotations
    ├── frames                          # Directory for storing frames
    └── annotations                     # Directory for storing annotations
```
![image](https://github.com/clenser/RoboVision/assets/100501976/1fb4cb3d-5092-4573-a187-ba358ec3b574)

![image](https://github.com/clenser/RoboVision/assets/100501976/223fa2b5-0a5c-435c-ab5a-4b4df539ea4c)

![image](https://github.com/clenser/RoboVision/assets/100501976/09d9de1b-b6ec-4f21-b752-ba8554911529)
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides an overview of the project, its features, setup instructions, and usage guidelines. Feel free to customize it further based on your specific requirements and project structure.
