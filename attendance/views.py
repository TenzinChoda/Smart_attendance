from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from .forms import StudentRegistrationForm
import cv2
import numpy as np
import os
import shutil
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
import json
from datetime import datetime
import csv
import base64
import pandas as pd

# Initialize face detector
detector = MTCNN()
IMG_SIZE = (64, 64)
CAPTURE_COUNT = 10  # Changed from 50 to 10

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame

def register_student(request):
    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            roll_number = form.cleaned_data['roll_number']
            
            # Create directory for student if it doesn't exist
            student_dir = os.path.join('attendance/static/captured_images', name)
            if not os.path.exists(student_dir):
                os.makedirs(student_dir)
                
            # Log registration
            log_registration(name, roll_number)
            
            return render(request, 'attendance/capture.html', {
                'name': name,
                'roll_number': roll_number
            })
    else:
        form = StudentRegistrationForm()
    
    return render(request, 'attendance/register.html', {'form': form})

def log_registration(name, roll_number):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists('registrations.csv'):
        with open('registrations.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Roll Number', 'Registration Date'])
    
    with open('registrations.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, roll_number, timestamp])

def view_registrations(request):
    try:
        df = pd.read_csv('registrations.csv')
        registrations = df.to_dict('records')
    except:
        registrations = []
    
    return render(request, 'attendance/registrations.html', {
        'registrations': registrations
    })

def delete_registration(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        
        try:
            # Remove from registrations.csv
            df = pd.read_csv('registrations.csv')
            df = df[df['Name'] != name]
            df.to_csv('registrations.csv', index=False)
            
            # Remove student's image directory
            student_dir = os.path.join('attendance/static/captured_images', name)
            if os.path.exists(student_dir):
                shutil.rmtree(student_dir)
            
            # Remove from class_labels.json if exists
            model_labels_path = 'attendance/models/class_labels.json'
            if os.path.exists(model_labels_path):
                with open(model_labels_path, 'r') as f:
                    labels = json.load(f)
                if name in labels:
                    labels.remove(name)
                with open(model_labels_path, 'w') as f:
                    json.dump(labels, f)
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def capture_frame(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        count = int(request.POST.get('count', 0))
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        cap.release()
        
        if success:
            # Detect face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame)
            
            if faces:
                x, y, w, h = faces[0]['box']
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, IMG_SIZE)
                
                # Save the image
                filename = f"{name}_{count}.jpg"
                filepath = os.path.join('attendance/static/captured_images', name, filename)
                cv2.imwrite(filepath, face)
                
                # Convert frame to base64 for preview
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                return JsonResponse({
                    'status': 'success',
                    'image': f'data:image/jpeg;base64,{frame_b64}',
                    'face_detected': True
                })
            
            return JsonResponse({
                'status': 'error',
                'message': 'No face detected'
            })
        
        return JsonResponse({
            'status': 'error',
            'message': 'Failed to capture image'
        })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def train_model(request):
    # Get all student directories
    base_dir = 'attendance/static/captured_images'
    students = os.listdir(base_dir)
    
    if not students:
        return JsonResponse({'status': 'error', 'message': 'No student data available'})
    
    # Prepare training data
    X = []
    y = []
    
    for idx, student in enumerate(students):
        student_dir = os.path.join(base_dir, student)
        for img_name in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face
            faces = detector.detect_faces(img)
            if faces:
                x, y_, w, h = faces[0]['box']
                face = img[y_:y_+h, x:x+w]
                face = cv2.resize(face, IMG_SIZE)
                face = img_to_array(face) / 255.0
                
                X.append(face)
                y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Create and train the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(students), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    
    # Save the model and class labels
    model.save('attendance/models/face_model.keras')
    with open('attendance/models/class_labels.json', 'w') as f:
        json.dump(students, f)
    
    return JsonResponse({'status': 'success'})

def gen_frames():
    camera = VideoCamera()
    model = None
    class_labels = None
    
    try:
        model = load_model('attendance/models/face_model.keras')
        with open('attendance/models/class_labels.json', 'r') as f:
            class_labels = json.load(f)
    except:
        pass
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
            
        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        
        for face in faces:
            x, y, w, h = face['box']
            face_img = rgb_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)
            face_array = img_to_array(face_img) / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            if model is not None and class_labels is not None:
                predictions = model.predict(face_array)
                class_idx = np.argmax(predictions)
                confidence = np.max(predictions)
                
                if confidence > 0.95:
                    name = class_labels[class_idx]
                    # Log attendance
                    log_attendance(name)
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (0, 255, 0), 2)
            else:
                # Just draw rectangle for face detection
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def log_attendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Read existing attendance
        df = pd.read_csv('attendance.csv', names=['Name', 'Timestamp'])
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d')
        
        # Check if student already has attendance for today
        if not df[(df['Name'] == name) & (df['Date'] == today)].empty:
            return  # Student already has attendance for today
        
        # Add new attendance
        with open('attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])
    except FileNotFoundError:
        # If file doesn't exist, create it and add the attendance
        with open('attendance.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp])

def delete_attendance(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        timestamp = request.POST.get('timestamp')
        
        try:
            # Read the current CSV file
            df = pd.read_csv('attendance.csv', names=['Name', 'Timestamp'])
            
            # Filter out the record to delete
            df = df[~((df['Name'] == name) & (df['Timestamp'] == timestamp))]
            
            # Write back to CSV
            df.to_csv('attendance.csv', index=False, header=False)
            
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def view_attendance(request):
    # Read the attendance CSV file
    try:
        df = pd.read_csv('attendance.csv', names=['Name', 'Timestamp'])
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        # Sort by timestamp in descending order
        df = df.sort_values('Timestamp', ascending=False)
        # Format timestamp for display
        df['Original_Timestamp'] = df['Timestamp']  # Keep original for deletion
        df['Timestamp'] = df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Convert to list of dictionaries for template
        attendance_logs = df.to_dict('records')
    except:
        attendance_logs = []
    
    return render(request, 'attendance/attendance_logs.html', {
        'attendance_logs': attendance_logs
    }) 