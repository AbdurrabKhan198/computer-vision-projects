"""
Hand Gesture Detection using OpenCV and MediaPipe
This script detects hand gestures in real-time using a webcam.
"""

import cv2
import mediapipe as mp
import numpy as np

class HandGestureDetector:
    def __init__(self):
        # Initialize MediaPipe Hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def count_fingers(self, hand_landmarks, handedness):
        """
        Count the number of extended fingers
        """
        # Finger tip and pip landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_pips = [2, 6, 10, 14, 18]
        
        fingers_up = []
        
        # Check thumb (different logic for left/right hand)
        if handedness == "Right":
            if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        else:
            if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Check other four fingers
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        return fingers_up.count(1)
    
    def detect_gesture(self, finger_count):
        """
        Detect specific gestures based on finger count
        """
        gestures = {
            0: "Fist",
            1: "One",
            2: "Peace/Two",
            3: "Three",
            4: "Four",
            5: "Open Hand/Five"
        }
        return gestures.get(finger_count, "Unknown")
    
    def process_frame(self, frame):
        """
        Process a single frame for hand detection
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        gesture_text = "No hand detected"
        
        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get handedness (Left or Right)
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Count fingers
                finger_count = self.count_fingers(hand_landmarks, handedness)
                
                # Detect gesture
                gesture = self.detect_gesture(finger_count)
                gesture_text = f"{handedness} Hand: {gesture} ({finger_count} fingers)"
                
                # Display gesture on frame
                cv2.putText(frame, gesture_text, (10, 50 + idx * 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, gesture_text
    
    def run(self):
        """
        Main loop to capture video and detect gestures
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Hand Gesture Detection Started!")
        print("Press 'q' to quit")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, gesture = self.process_frame(frame)
            
            # Add instructions
            cv2.putText(processed_frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow("Hand Gesture Detection", processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Hand Gesture Detection Stopped!")


def main():
    """
    Main function to run the hand gesture detector
    """
    detector = HandGestureDetector()
    detector.run()


if __name__ == "__main__":
    main()
