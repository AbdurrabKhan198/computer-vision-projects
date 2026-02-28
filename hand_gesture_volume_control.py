"""
Hand Gesture Volume Control using OpenCV and MediaPipe
This script controls your computer's volume using hand gestures in real-time.
- Use thumb and index finger distance to control volume
- Pinch fingers together for minimum volume
- Spread fingers apart for maximum volume
"""

import cv2
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER


class HandGestureVolumeControl:
    def __init__(self):
        # Initialize MediaPipe Hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only need one hand for volume control
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize volume control
        self.setup_volume_control()
        
        # Volume bar parameters
        self.vol_bar_x = 50
        self.vol_bar_y = 150
        self.vol_bar_height = 300
        self.vol_bar_width = 50
        
    def setup_volume_control(self):
        """
        Setup Windows volume control interface
        """
        try:
            # Get all audio devices
            devices = AudioUtilities.GetSpeakers()
            
            # Activate the IAudioEndpointVolume interface
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume range
            vol_range = self.volume.GetVolumeRange()
            self.min_vol = vol_range[0]
            self.max_vol = vol_range[1]
            
            print(f"✓ Volume control initialized successfully!")
            print(f"  Volume range: {self.min_vol:.2f} dB to {self.max_vol:.2f} dB")
            
        except AttributeError as e:
            print(f"✗ AttributeError: {e}")
            print("  Trying alternative method...")
            self._setup_volume_alternative()
            
        except Exception as e:
            print(f"✗ Error initializing volume control: {e}")
            print("  Volume control disabled. Visualization only mode.")
            self.volume = None
            self.min_vol = -65.25
            self.max_vol = 0.0
    
    def _setup_volume_alternative(self):
        """
        Alternative setup method for volume control
        """
        try:
            from pycaw.utils import AudioUtilities
            from pycaw.api.endpointvolume import IAudioEndpointVolume
            
            # Get default audio endpoint
            devices = AudioUtilities.GetSpeakers()
            
            # Get the audio endpoint volume interface
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume range
            vol_range = self.volume.GetVolumeRange()
            self.min_vol = vol_range[0]
            self.max_vol = vol_range[1]
            
            print(f"✓ Volume control initialized (alternative method)!")
            print(f"  Volume range: {self.min_vol:.2f} dB to {self.max_vol:.2f} dB")
            
        except Exception as e:
            print(f"✗ Alternative method also failed: {e}")
            print("  Volume control disabled. Visualization only mode.")
            self.volume = None
            self.min_vol = -65.25
            self.max_vol = 0.0
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    def process_frame(self, frame):
        """
        Process a single frame for hand detection and volume control
        """
        h, w, c = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Default values
        length = 0
        vol_percentage = 0
        
        # Draw hand landmarks and control volume
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get thumb tip (landmark 4) and index finger tip (landmark 8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                # Convert normalized coordinates to pixel coordinates
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Calculate midpoint
                mid_x, mid_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2
                
                # Draw circles on thumb and index finger tips
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
                cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 0), cv2.FILLED)
                
                # Calculate distance between thumb and index finger
                length = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
                
                # Hand range: 20-200 pixels (adjust based on your camera distance)
                # Volume range: min_vol to max_vol dB
                min_length = 20
                max_length = 200
                
                # Clamp length to valid range
                length = max(min_length, min(length, max_length))
                
                # Convert length to volume
                vol = np.interp(length, [min_length, max_length], [self.min_vol, self.max_vol])
                vol_percentage = np.interp(length, [min_length, max_length], [0, 100])
                
                # Set volume (only if volume control is available)
                if self.volume:
                    try:
                        self.volume.SetMasterVolumeLevel(vol, None)
                    except Exception as e:
                        print(f"Error setting volume: {e}")
                
                # Change circle color when fingers are very close (muted)
                if length < 30:
                    cv2.circle(frame, (mid_x, mid_y), 12, (0, 0, 255), cv2.FILLED)
        
        # Draw volume bar
        self.draw_volume_bar(frame, vol_percentage)
        
        # Display volume percentage
        status_text = f"Volume: {int(vol_percentage)}%"
        if not self.volume:
            status_text += " (Visualization Only)"
        
        cv2.putText(frame, status_text, 
                   (self.vol_bar_x - 10, self.vol_bar_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def draw_volume_bar(self, frame, vol_percentage):
        """
        Draw a visual volume bar on the frame
        """
        # Draw outer rectangle (background)
        cv2.rectangle(frame, 
                     (self.vol_bar_x, self.vol_bar_y), 
                     (self.vol_bar_x + self.vol_bar_width, self.vol_bar_y + self.vol_bar_height),
                     (255, 255, 255), 3)
        
        # Calculate filled height based on volume percentage
        filled_height = int(np.interp(vol_percentage, [0, 100], [self.vol_bar_height, 0]))
        
        # Draw filled rectangle (volume level)
        cv2.rectangle(frame,
                     (self.vol_bar_x, self.vol_bar_y + filled_height),
                     (self.vol_bar_x + self.vol_bar_width, self.vol_bar_y + self.vol_bar_height),
                     (0, 255, 0), cv2.FILLED)
        
        # Add percentage markers
        for i in range(0, 101, 25):
            marker_y = self.vol_bar_y + int(np.interp(i, [0, 100], [self.vol_bar_height, 0]))
            cv2.line(frame, 
                    (self.vol_bar_x + self.vol_bar_width, marker_y),
                    (self.vol_bar_x + self.vol_bar_width + 10, marker_y),
                    (255, 255, 255), 2)
            cv2.putText(frame, f"{i}", 
                       (self.vol_bar_x + self.vol_bar_width + 15, marker_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """
        Main loop to capture video and control volume
        """
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("🎵 Hand Gesture Volume Control Started! 🎵")
        print("="*60)
        print("\nInstructions:")
        print("  👉 Pinch thumb and index finger together → Decrease volume")
        print("  👉 Spread thumb and index finger apart → Increase volume")
        print("  👉 Press 'q' to quit")
        print("="*60 + "\n")
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Add instructions
            cv2.putText(processed_frame, "Pinch to decrease, Spread to increase", 
                       (150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Press 'q' to quit", 
                       (150, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow("Hand Gesture Volume Control", processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n" + "="*60)
        print("🛑 Hand Gesture Volume Control Stopped!")
        print("="*60)


def main():
    """
    Main function to run the hand gesture volume control
    """
    controller = HandGestureVolumeControl()
    controller.run()


if __name__ == "__main__":
    main()
