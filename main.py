import cv2
import numpy as np
from datetime import datetime
import os
import pickle
import json
import logging
from threading import Timer
import sqlite3

class EnhancedFaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Initialize database
        self.init_database()
        
        # Load existing data
        self.known_faces = self.load_known_faces()
        self.face_id = len(self.known_faces)
        
        # Configure logging
        logging.basicConfig(filename='face_recognition.log', level=logging.INFO)
        
    def init_database(self):
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS access_logs
                    (timestamp TEXT, name TEXT, confidence REAL)''')
        conn.commit()
        conn.close()

    def log_access(self, name, confidence):
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO access_logs VALUES (?, ?, ?)", 
                 (timestamp, name, confidence))
        conn.commit()
        conn.close()
        logging.info(f"Access logged: {name} at {timestamp} with confidence {confidence}")

    def load_known_faces(self):
        if os.path.exists('known_faces.pkl'):
            with open('known_faces.pkl', 'rb') as f:
                return pickle.load(f)
        return {}

    def save_known_faces(self):
        with open('known_faces.pkl', 'wb') as f:
            pickle.dump(self.known_faces, f)

    def register_face(self, name, required_samples=50):
        cap = cv2.VideoCapture(0)
        face_samples = []
        count = 0
        
        while count < required_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Check for eyes to ensure face is real
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2:  # Only accept faces with both eyes visible
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_samples.append(gray[y:y+h, x:x+w])
                    count += 1
                    
                    # Progress indicator
                    progress = f"Capturing: {count}/{required_samples}"
                    cv2.putText(frame, progress, (10, 30), self.font, 1, (0, 255, 0), 2)
            
            cv2.imshow('Registering Face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) >= required_samples:
            labels = [self.face_id] * len(face_samples)
            self.known_faces[self.face_id] = {
                'name': name,
                'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.recognizer.train(face_samples, np.array(labels))
            self.save_known_faces()
            self.face_id += 1
            logging.info(f"New face registered: {name}")
            return True
        return False

    def generate_report(self):
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        
        # Get today's logs
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute("""
            SELECT name, COUNT(*), AVG(confidence)
            FROM access_logs
            WHERE timestamp LIKE ?
            GROUP BY name
        """, (f"{today}%",))
        
        report = {
            'date': today,
            'total_accesses': 0,
            'unique_visitors': 0,
            'details': []
        }
        
        for row in c.fetchall():
            report['details'].append({
                'name': row[0],
                'access_count': row[1],
                'avg_confidence': round(row[2], 2)
            })
            report['total_accesses'] += row[1]
            report['unique_visitors'] += 1
            
        conn.close()
        
        # Save report
        with open(f'report_{today}.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        return report


    def start_recognition(self, confidence_threshold=50):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) >= 2:  # Verify real face with eye detection
                    try:
                        id_, conf = self.recognizer.predict(roi_gray)
                        confidence = 100 - conf
                        
                        if confidence >= confidence_threshold:
                            name = self.known_faces[id_]['name']
                            color = (0, 255, 0)  # Green for high confidence
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)  # Red for low confidence
                            
                        # Log access
                        if name != "Unknown":
                            self.log_access(name, confidence)
                            
                        # Draw rectangle and labels
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, name, (x, y-10), self.font, 0.9, color, 2)
                        cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                                  (x, y+h+25), self.font, 0.7, color, 2)
                    except Exception as e:
                        logging.error(f"Recognition error: {str(e)}")
            
            # Display current time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, current_time, (10, 30), self.font, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):  # Generate report
                report = self.generate_report()
                print("Report generated:", report)
                
        cap.release()
        cv2.destroyAllWindows()

# Usage Example
if __name__ == "__main__":
    face_system = EnhancedFaceRecognitionSystem()
    
    # Register new face
    print("Registering new face...")
    if face_system.register_face("Amit mehra"):
        print("Registration successful!")
    else:
        print("Registration failed!")
    
    # Start recognition
    print("Starting face recognition...")
    print("Press 'q' to quit, 'r' to generate report")
    face_system.start_recognition(confidence_threshold=60)