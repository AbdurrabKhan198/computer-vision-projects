"""
Hand Gesture Detection using OpenCV and cvzone
This script detects hand gestures in real-time using a webcam.
A simpler alternative that avoids MediaPipe dependency conflicts.
"""

import cv2
try:
    from cvzone.HandTrackingModule import HandDetector
except ImportError:
    print("Installing required package: cvzone...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cvzone"])
    from cvzone.HandTrackingModule import HandDetector

class HandGestureDetector:
    def __init__(self):
        # Initialize cvzone Hand Detector
        self.detector = HandDetector(detectionCon=0.7, maxHands=2)
        
    def detect_gesture(self, finger_count):
        """
        Detect specific gestures based on finger count
        """
        gestures = {
            0: "Fist ✊",
            1: "One ☝️",
            2: "Peace/Two ✌️",
            3: "Three 🤟",
            4: "Four 🖖",
            5: "Open Hand/Five ✋"
        }
        return gestures.get(finger_count, "Unknown")
    
    def run(self):
        """
        Main loop to capture video and detect gestures
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)  # Width
        cap.set(4, 720)   # Height
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("=" * 50)
        print("Hand Gesture Detection Started!")
        print("=" * 50)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Show different hand gestures to the camera")
        print("=" * 50)
        
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            
            if not success:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            hands, frame = self.detector.findHands(frame, flipType=False)
            
            # Process detected hands
            if hands:
                for idx, hand in enumerate(hands):
                    # Get hand information
                    handType = hand["type"]  # "Left" or "Right"
                    fingers = self.detector.fingersUp(hand)
                    finger_count = fingers.count(1)
                    
                    # Detect gesture
                    gesture = self.detect_gesture(finger_count)
                    
                    # Get hand center position
                    lmList = hand["lmList"]
                    x, y = hand["center"]
                    
                    # Display gesture text near hand
                    text = f"{handType}: {gesture}"
                    cv2.putText(frame, text, (x - 100, y - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display finger count
                    cv2.putText(frame, f"Fingers: {finger_count}", (x - 100, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add instructions at bottom
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow("Hand Gesture Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 50)
        print("Hand Gesture Detection Stopped!")
        print("=" * 50)


def main():
    """
    Main function to run the hand gesture detector
    """
    try:
        detector = HandGestureDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have a webcam connected and accessible.")


if __name__ == "__main__":
    main()
