import cv2
import numpy as np
import os
import pickle
from pathlib import Path

class FaceSecuritySystem:
    def __init__(self):
        self.encoding_file = "face_encoding.pkl"
        self.known_face_encoding = None
        
        # Load OpenCV's pre-trained cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.load_face_encoding()
        
    def load_face_encoding(self):
        """Load stored face encoding if it exists"""
        if os.path.exists(self.encoding_file):
            with open(self.encoding_file, 'rb') as f:
                self.known_face_encoding = pickle.load(f)
            print("✓ Face password loaded successfully!")
        else:
            print("No face password found. Please enroll first.")
    
    def save_face_encoding(self, encoding):
        """Save face encoding to file"""
        with open(self.encoding_file, 'wb') as f:
            pickle.dump(encoding, f)
        print("✓ Face password saved successfully!")
    
    def extract_face_features(self, face_region):
        """Extract facial features from a face region using histogram"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram of face region
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Extract more features: face shape and edges
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        
        # Combine features
        features = np.concatenate([hist, edge_hist])
        
        # Add face dimensions as features
        h, w = gray.shape
        features = np.append(features, [h, w])
        
        return features
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        return faces
    
    def compare_features(self, feature1, feature2):
        """Compare two feature vectors using correlation coefficient"""
        if len(feature1) != len(feature2):
            return float('inf')
        
        # Calculate normalized Euclidean distance
        distance = np.linalg.norm(feature1 - feature2)
        return distance / (len(feature1) ** 0.5)  # Normalize by feature count
    
    def enroll_face(self):
        """Capture and store face encoding for enrollment"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return False
        
        captured_encoding = None
        enrollment_complete = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if not enrollment_complete:
                # Draw center box
                cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
                cv2.putText(frame, "ENROLLMENT MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(frame, "Position face in frame", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, "Press SPACE to capture | Q to quit", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                face_count = len(faces)
                
                # Draw rectangles around detected faces
                for (x, y, w_face, h_face) in faces:
                    cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
                
                # Display face count
                if face_count == 0:
                    cv2.putText(frame, "No face detected", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif face_count > 1:
                    cv2.putText(frame, f"Multiple faces detected: {face_count}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Face detected - Press SPACE to capture", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Face Enrollment', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' ') and face_count == 1:  # Space to capture
                    # Extract face features
                    x, y, w_face, h_face = faces[0]
                    face_region = frame[y:y+h_face, x:x+w_face]
                    
                    features = self.extract_face_features(face_region)
                    captured_encoding = features
                    
                    # Draw success feedback
                    cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 3)
                    cv2.putText(frame, "FACE CAPTURED!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, "Saved! Press SPACE to verify", (10, h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Face Enrollment', frame)
                    cv2.waitKey(1500)
                    
                    # Save the encoding
                    self.known_face_encoding = captured_encoding
                    self.save_face_encoding(captured_encoding)
                    enrollment_complete = True
                
                elif key == ord('q'):  # Q to quit
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            else:
                # Show enrollment complete screen
                cv2.putText(frame, "ENROLLMENT COMPLETE!", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, "Face password saved successfully!", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to verify your face", (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(frame, "Press Q to return to menu", (20, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                cv2.imshow('Face Enrollment', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space to verify
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
                elif key == ord('q'):  # Q to return to menu
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    def authenticate_face(self):
        """Authenticate user by comparing with stored face"""
        if self.known_face_encoding is None:
            # Show error on screen
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    # Semi-transparent overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    cv2.putText(frame, "NO FACE PASSWORD FOUND!", (w//4 + 20, h//2 - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(frame, "Please enroll first", (w//4 + 50, h//2 + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Press any key to continue", (w//4 + 40, h//2 + 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    cv2.imshow('Face Authentication', frame)
                    cv2.waitKey(3000)
                
                cap.release()
            cv2.destroyAllWindows()
            return False
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return False
        
        auth_attempts = 0
        max_attempts = 3
        match_threshold = 0.8  # Similarity threshold (lower = stricter)
        frame_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_counter += 1
            
            # Display header
            cv2.putText(frame, "AUTHENTICATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.2, (0, 255, 255), 2)
            cv2.putText(frame, f"Attempts: {auth_attempts}/{max_attempts}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to cancel", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                # No face detected
                cv2.putText(frame, "No face detected", (w//2 - 150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                cv2.putText(frame, "Move closer to camera", (w//2 - 150, h//2 + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            
            else:
                for (x, y, w_face, h_face) in faces:
                    # Extract features from current frame
                    face_region = frame[y:y+h_face, x:x+w_face]
                    current_features = self.extract_face_features(face_region)
                    
                    # Compare with known face
                    distance = self.compare_features(self.known_face_encoding, current_features)
                    similarity_score = max(0, (1 - distance / 2))  # Convert distance to similarity score
                    
                    # Determine match
                    is_match = distance < match_threshold
                    
                    if is_match:
                        # GREEN for match - PASSED
                        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 4)
                        cv2.putText(frame, "PASSED", (x + 10, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                        cv2.putText(frame, f"Match score: {similarity_score:.1%}", (x, y + h_face + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Success overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h//5), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        
                        cv2.putText(frame, "AUTHENTICATION SUCCESSFUL!", (30, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                        
                        cv2.imshow('Face Authentication', frame)
                        cv2.waitKey(2000)
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                    
                    else:
                        # RED for no match - FAILED
                        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 0, 255), 4)
                        cv2.putText(frame, "FAILED", (x + 10, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        cv2.putText(frame, f"Distance: {distance:.2f}", (x, y + h_face + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        auth_attempts += 1
                        
                        if auth_attempts >= max_attempts:
                            # Lockout screen
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                            
                            cv2.putText(frame, "AUTHENTICATION FAILED!", (w//2 - 200, h//2 - 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            cv2.putText(frame, "Maximum attempts exceeded", (w//2 - 200, h//2 + 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(frame, "Access denied!", (w//2 - 100, h//2 + 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                            
                            cv2.imshow('Face Authentication', frame)
                            cv2.waitKey(2500)
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
            
            cv2.imshow('Face Authentication', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        return False


def main():
    print("=" * 50)
    print("  FACE RECOGNITION SECURITY SYSTEM")
    print("=" * 50)
    
    system = FaceSecuritySystem()
    
    while True:
        print("\n" + "=" * 50)
        print("Main Menu:")
        print("1. Enroll Face (Set Face Password)")
        print("2. Authenticate (Verify Face)")
        print("3. Exit")
        print("=" * 50)
        
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == '1':
            if system.enroll_face():
                # After enrollment complete, automatically verify
                print("\nVerifying face password...")
                if system.authenticate_face():
                    # Show success on console too
                    print("\n✓ Enrollment and verification successful!")
                else:
                    print("\n⚠ Enrollment saved but verification failed. Try authenticating again.")
            else:
                print("\n✗ Enrollment cancelled.")
        
        elif choice == '2':
            result = system.authenticate_face()
            if result:
                print("\n✓ Access granted!")
            else:
                print("\n✗ Access denied!")
        
        elif choice == '3':
            print("\nGoodbye!")
            break
        
        else:
            print("\n❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()

#Gopolang Rodney Diutlwileng