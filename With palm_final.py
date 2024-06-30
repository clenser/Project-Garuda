import cv2
import mediapipe as mp
import math
import numpy as np
from deepface import DeepFace
import sys
import pyfirmata2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bottle import route, run, response
import datetime
import os
import json
import firebase_admin
from firebase_admin import credentials, storage
import datetime
from tqdm import tqdm  
import socket
import shutil
import face_recognition
import tkinter as tk
from tkinter import simpledialog, messagebox
import requests
#########################################################################################################
#integrating callhimp
url = "https://api.callchimp.ai/v1/calls"  
data = {
    "lead": "837496"
}

headers = {
    "content-type": "application/json",
    "x-api-key":'uWCv5aSv.ptSy0eQc4ZXLdfBVOURMDN8UBxwe66Eo'
}


#Get current Date and time
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#Initializing Global ariable
user=0
####################################################################################################################################
#palm detection
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
map_range = lambda x, in_min, in_max, out_min, out_max: abs(
    (x - in_min) * (out_max - out_min) // (in_max - in_min)
)
min_palm_size = 0.08
max_palm_size = 0.25

#Firebase Data logging
base_dir = r"C:\Users\srbar\OneDrive\Desktop\New folder\Image_annotations"
frame_dir = os.path.join(base_dir, "frames")
annotation_dir = os.path.join(base_dir, "annotations")
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)

def upload_images_to_firebase(folder_path_annotations, folder_path_frames):
    # Constants for Firebase credentials and bucket name
    service_account_key_path = 'C:\\Users\\srbar\\OneDrive\\Desktop\\New folder\\serviceAccountKey.json'
    bucket_name = 'robovision-fd6e2.appspot.com'

    # Initialize Firebase with your service account credentials
    cred = credentials.Certificate(service_account_key_path)
    firebase_admin.initialize_app(cred, {
        'storageBucket': bucket_name
    })

    # Reference to the Firebase Storage bucket
    bucket = storage.bucket()

    uploaded_urls = {}

    # Get current date and time
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Create folder structure based on user_name and current date-time
    user_folder = f'{user_name}/'
    datetime_folder = f'{now}/'
    annotations_folder = 'annotations/'
    frames_folder = 'captured_frames/'

    # Upload annotations
    annotations_path = folder_path_annotations
    annotations_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
    progress_bar_annotations = tqdm(annotations_files, desc='Uploading Annotations', unit='file')
    for filename in progress_bar_annotations:
        local_file_path = os.path.join(annotations_path, filename)
        destination_blob_name = user_folder + datetime_folder + annotations_folder + filename
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        download_url = blob.generate_signed_url(expiration=3600)
        uploaded_urls[filename] = download_url
        progress_bar_annotations.set_postfix({'Uploaded': filename})

    # Upload captured frames
    frames_path = folder_path_frames
    frames_files = [f for f in os.listdir(frames_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    progress_bar_frames = tqdm(frames_files, desc='Uploading Frames', unit='file')
    for filename in progress_bar_frames:
        local_file_path = os.path.join(frames_path, filename)
        destination_blob_name = user_folder + datetime_folder + frames_folder + filename
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        download_url = blob.generate_signed_url(expiration=3600)
        uploaded_urls[filename] = download_url
        progress_bar_frames.set_postfix({'Uploaded': filename})

    return uploaded_urls

# Paths to the annotations and frames folders
folder_path_annotations = r'C:\Users\srbar\OneDrive\Desktop\New folder\Image_annotations\annotations'
folder_path_frames = r'C:\Users\srbar\OneDrive\Desktop\New folder\Image_annotations\frames'

#video stream
def start_video_stream():
    
    cap = cv2.VideoCapture(0)

    @route('/video_feed')
    def video_feed():
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield frame in HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Get the IP address of the machine dynamically
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # Construct and print the URL
    stream_url = f"http://{local_ip}:8080/video_feed"
    
    # Upload images to Firebase Storage and get their download URLs
   
    print(f"Streaming URL: {stream_url}")

    # Start Bottle server
    run(host='0.0.0.0', port=8080)
    # upload_images_to_firebase(folder_path_annotations, folder_path_frames)
    print("Data Uploaded to cloud successfully")
    
      
#For email
def send_email(subject, body, to_email, from_email, password):
    try:
        # Setup the MIME
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = subject

        # Attach the body with the msg instance
        message.attach(MIMEText(body, 'plain'))

        # Create SMTP session for sending the mail
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable security
        server.login(from_email, password)  # Login with your email and password
        text = message.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
    except Exception as e:
        print(f'Failed to send email: {str(e)}')
        
#Distance Calculation    
def calculate_distance(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    distance = math.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2)
    return round(distance, 1)
        
#emotion analysis

def analyze_emotion(image):
    try:
        predictions = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return max(predictions[0]['emotion'], key=predictions[0]['emotion'].get).title()
    except Exception as e:
        print("Error in emotion analysis:", e)
        return None
    
def recognize_face(image, user_data):
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for face_encoding in face_encodings:
            for user_name, user_info in user_data.items():
                for face_path in user_info['face_paths']:
                    db_image = face_recognition.load_image_file(face_path)
                    db_face_encoding = face_recognition.face_encodings(db_image)[0]
                    matches = face_recognition.compare_faces([db_face_encoding], face_encoding)
                    if True in matches:
                        return user_name
        return None
    except Exception as e:
        print("Error in face recognition:", e)
        return None

user_data = {
    'Sreemanta': {
        'face_paths': ["C:/Users/srbar/Downloads/Telegram Desktop/sreemanta{}.JPG".format(i) for i in range(1, 9)],
        'secret_key': "1234"
    },
}
# Tkinter GUI for user prompts and PIN code input
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("User Verification")
        self.geometry("300x150")
        self.label = tk.Label(self, text="Initializing...", wraplength=250)
        self.label.pack(pady=20)
        self.pin_code = None

    def update_label(self, message):
        self.label.config(text=message)

    def get_pin_code(self, prompt):
        self.label.config(text=prompt)
        self.pin_code = simpledialog.askstring("PIN Code", "Enter your PIN code:", show='*')
        return self.pin_code

    def show_message(self, title, message):
        messagebox.showinfo(title, message)
        
# Function to start the Tkinter application
def start_tkinter_app():
    app = App()
    app.update_label("Capturing initial image for emotion analysis and face recognition...")
    app.update()
    return app
cap = cv2.VideoCapture(0)
app = start_tkinter_app()

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()
    
    
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image.")
    else:
        captured_image = frame.copy()
        detected_emotion = analyze_emotion(captured_image)
        if detected_emotion and detected_emotion in ["Happy", "Neutral", "Surprise","Fear"]:
            app.update_label(f"2. Detected Emotion: {detected_emotion}")
            app.update()
            user_name = recognize_face(captured_image, user_data)
            if user_name:
                user=user_name
                app.update_label(f"3. Emotion is allowed. {user_name}'s face recognized.")
                app.update()
                secret_key = app.get_pin_code(f"Enter the secret key for {user_name}: ")
                if secret_key == user_data[user_name]['secret_key']:
                    app.update_label(f"4. {user_name}, you are in control.")
                    app.update()
                    app.show_message("Access Granted", f"Welcome {user_name}! You are now in control.")
                    send_email(
                            subject='New Login Detected!',
                            body=f'This is to inform the host that the user {user_name} just logged into the system with their passcode.The date and time are as follows :{now}',
                            to_email='srbarman00@gmail.com',
                            from_email='circuitcogs@gmail.com',
                            password='ixer eztr sjmr sihe'
                        )
                    cap.release()
                else:
                    app.update_label(f"Incorrect PIN code for {user_name}.")
                    app.show_message("Access Denied", "Incorrect PIN code.")
                    sys.exit()
            else:
                app.update_label("Face not recognized.")
                app.show_message("Access Denied", "Face not recognized.")
                sys.exit()
        else:
            app.update_label("Emotion not allowed or not detected.")
            app.show_message("Access Denied", "Emotion not allowed or not detected.")
            sys.exit()

cap.release()
cv2.destroyAllWindows()

##########################################################################################################################################

# Initialize global variables
current_servo1_position = 90
current_servo2_position = 90
current_servo3_position = 90
current_servo4_position = 90
current_servo5_position = 90
steps = 20

board = pyfirmata2.Arduino('COM4')
servo_pin1 = board.get_pin('d:3:s')
servo_pin2 = board.get_pin('d:5:s')
servo_pin3 = board.get_pin('d:6:s')
servo_pin4 = board.get_pin('d:9:s')
servo_pin5 = board.get_pin('d:10:s')
servo_pin6 = board.get_pin('d:11:s')
servo_pin7 = board.get_pin('d:12:s')
servo_pin8 = board.get_pin('d:13:s')

servo_pin7.write(90)
servo_pin8.write(90)

def update_servos1(angle1, angle2):
    if angle1 >= 0 and angle1 <= 180:
        servo_pin1.write(angle1)
    if angle2 >= 0 and angle2 <= 180:   
        servo_pin2.write(angle2)

   
def update_servos2(angle3, angle4):

    if angle3 >= 0 and angle3 <= 180:   
        servo_pin3.write(angle3)
    if angle4 >= 0 and angle4 <= 180:  
        servo_pin4.write(angle4)

def update_servos3(angle5):
    if angle5 >= 0 and angle5 <= 180:   
        servo_pin5.write(angle5)

def update_servos4(angle6):
    if angle6 >= 0 and angle6 <= 180:   
        servo_pin6.write(angle6)

   
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = math.degrees(radians)
    if angle < 0:
        angle += 360
    return angle

def interpolate_smoothly(start_angle, end_angle, steps):
    angles = np.linspace(start_angle, end_angle, steps)
    smoothed_angles = np.sin(np.linspace(0, np.pi, steps)) * (end_angle - start_angle) / 2 + (start_angle + end_angle) / 2
    return smoothed_angles

#adding face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing_face = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 700
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0
    c=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)
        
        if face_results.detections:
            # Take the first face detected
            detection = face_results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            # Draw rectangle around the face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
            
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            
            #####################################################################
            frame_path = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            annotation_path = os.path.join(annotation_dir, f"annotation_{frame_count:04d}.json")
            landmarks = [{
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            } for lm in results.pose_landmarks.landmark]
                
            with open(annotation_path, "w") as f:
                json.dump(landmarks, f, indent=4)
            frame_count += 1

            #####################################################################

                
            # Landmark indices for connecting lines
            connections = [
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP), 
                (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),   
                # Add more connections as needed
            ]
            
            # Draw lines between connected landmarks
            for connection in connections:
                landmark1 = pose_landmarks[connection[0].value]
                landmark2 = pose_landmarks[connection[1].value]
                if landmark1.visibility > 0 and landmark2.visibility > 0:
                    landmark1_px = (int(landmark1.x * WINDOW_WIDTH), int(landmark1.y * WINDOW_HEIGHT))
                    landmark2_px = (int(landmark2.x * WINDOW_WIDTH), int(landmark2.y * WINDOW_HEIGHT))
                    cv2.line(frame, landmark1_px, landmark2_px, (255, 0, 0), 2)
            
            left_shoulder = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * WINDOW_WIDTH),
                            int(pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * WINDOW_HEIGHT))
            left_hip = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * WINDOW_HEIGHT))
            left_elbow = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * WINDOW_HEIGHT))
            left_wrist = (int(pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * WINDOW_HEIGHT))
            
            right_shoulder = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * WINDOW_WIDTH),
                            int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * WINDOW_HEIGHT))
            right_hip = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * WINDOW_HEIGHT))
            right_elbow = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * WINDOW_HEIGHT))
            right_wrist = (int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * WINDOW_WIDTH),
                        int(pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * WINDOW_HEIGHT))
            
            # For left arm
            servo1 = (calculate_angle(left_hip, left_shoulder, left_elbow) - 356) * -1
            servo2 = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            servo1_smoothed = interpolate_smoothly(current_servo1_position, servo1, steps)
            servo2_smoothed = interpolate_smoothly(current_servo2_position, servo2, steps)
            
            update_servos1(servo1_smoothed[-1],servo2_smoothed[-1])
            
            current_servo1_position = int(servo1_smoothed[-1])
            current_servo2_position = int(servo2_smoothed[-1])
            
            # For right arm
            servo4 = (calculate_angle(right_hip, right_shoulder, right_elbow) ) 
            servo5 = (calculate_angle(right_shoulder, right_elbow, right_wrist)-360 ) *-1
            
            servo4_smoothed = interpolate_smoothly(current_servo4_position, servo4, steps)
            servo5_smoothed = interpolate_smoothly(current_servo5_position, servo5, steps)
            
            update_servos2(servo4_smoothed[-1],servo5_smoothed[-1])
            
            current_servo4_position = int(servo4_smoothed[-1])
            current_servo5_position = int(servo5_smoothed[-1])
            

            cv2.circle(frame, left_shoulder, 5, (0, 0, 255), -1)
            cv2.circle(frame, left_hip, 5, (0, 0, 255), -1)
            cv2.circle(frame, left_elbow, 5, (0, 0, 255), -1)
            cv2.circle(frame, left_wrist, 5, (0, 0, 255), -1)

            cv2.putText(frame, f"Servo 1: {servo1:.2f} degrees", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Servo 2: {servo2:.2f} degrees", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            cv2.circle(frame, right_shoulder, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_hip, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_elbow, 5, (0, 0, 255), -1)
            cv2.circle(frame, right_wrist, 5, (0, 0, 255), -1)

            cv2.putText(frame, f"Servo 4: {servo4:.2f} degrees", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Servo 5: {servo5:.2f} degrees", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x > 0.5:
                    hand_landmark_list = hand_landmarks.landmark
                    if hand_landmark_list[mp_hands.HandLandmark.WRIST].x > hand_landmark_list[mp_hands.HandLandmark.THUMB_TIP].x:
                    # Calculate distance between index finger tip and thumb tip
                        distance = calculate_distance(hand_landmarks)
                        distance_str = "{:.1f}".format(distance)
                        #if distance == 0.0:
                            # send_email(
                            #     subject='Emergency! Danger Detected',
                            #     body='Please help me out. I am in danger.Follow the link http://192.168.1.8:8080/video_feed',
                            #     to_email='srbarman00@gmail.com',
                            #     from_email='circuitcogs@gmail.com',
                            #     password='ixer eztr sjmr sihe'
                            # )
                            # cap.release()
                            # response = requests.post(url, json=data,headers=headers)
                            # cv2.destroyAllWindows()
                            # start_video_stream()
                    # Right hand detected
                    right_middle_finger_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * WINDOW_WIDTH),                                      int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * WINDOW_HEIGHT))
                    right_wrist = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WINDOW_WIDTH),
                                int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * WINDOW_HEIGHT))
                    wrist_angle_R = calculate_angle(right_elbow, right_wrist, right_middle_finger_tip) 
                    
                    servo3= wrist_angle_R
                    servo3_smoothed = interpolate_smoothly(current_servo3_position, servo3, steps)
                    
                    update_servos3(servo3)
                    
                    current_servo3_position = int(servo3_smoothed[-1])

                    cv2.circle(frame, right_middle_finger_tip, 5, (0, 0, 255), -1)
                    cv2.circle(frame, right_wrist, 5, (0, 0, 255), -1)
                    cv2.line(frame, right_middle_finger_tip, right_wrist, (0, 255, 0), 2)  # Draw line in green
                    cv2.putText(frame, f"Servo 3: {wrist_angle_R:.2f} degrees", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Left hand detected
                    left_middle_finger_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * WINDOW_WIDTH),
                                            int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * WINDOW_HEIGHT))
                    left_wrist = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WINDOW_WIDTH),
                                int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * WINDOW_HEIGHT))
                    wrist_angle_L = calculate_angle(left_elbow, left_wrist, left_middle_finger_tip) - 120
                    
                    servo4= wrist_angle_L
                    
                    servo4_smoothed = interpolate_smoothly(current_servo4_position, servo4, steps)
                    
                    update_servos4(servo4)
                    
                    current_servo4_position = int(servo4_smoothed[-1])
                
                    cv2.circle(frame, left_middle_finger_tip, 5, (0, 0, 255), -1)
                    cv2.circle(frame, left_wrist, 5, (0, 0, 255), -1)
                    cv2.line(frame, left_middle_finger_tip, left_wrist, (0, 255, 0), 2)  # Draw line in green
                    cv2.putText(frame, f"Servo 6: {wrist_angle_L:.2f} degrees", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                handedness = hands_results.multi_handedness[idx].classification[0].label

                WRIST = hand_landmarks.landmark[0]
                INDEX_FINGER_MCP = hand_landmarks.landmark[5]
                palm_size = (
                    (WRIST.x - INDEX_FINGER_MCP.x) ** 2 +
                    (WRIST.y - INDEX_FINGER_MCP.y) ** 2 +
                    (WRIST.z - INDEX_FINGER_MCP.z) ** 2
                ) ** 0.5

                # Clamp the Palm Size using calibrated min and max values
                palm_size = clamp(palm_size, min_palm_size, max_palm_size)

                # Map Palm Size to Servo Angle Range
                z_min = 10
                z_max = 180
                z_angle = map_range(palm_size, min_palm_size, max_palm_size, z_max, z_min)
                
                # Annotate the image with palm size and angle
                if handedness == 'Right':
                    hand_label = 'Right Hand'
                    servo_pin7.write(z_angle)
                    cv2.putText(frame, f"{hand_label}: Palm Size: {palm_size:.2f}, Z Angle: {z_angle:.2f}", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    hand_label = 'Left Hand'
                    servo_pin8.write(z_angle)
                    cv2.putText(frame, f"{hand_label}: Palm Size: {palm_size:.2f}, Z Angle: {z_angle:.2f}", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

upload_images_to_firebase(folder_path_annotations, folder_path_frames)

# Print uploaded URLs for verification
print("Data Uploaded to cloud successfully")

shutil.rmtree(base_dir)
print('Process Completed Memory freed')
