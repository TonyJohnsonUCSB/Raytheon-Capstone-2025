from picamera2 import Picamera2
import cv2

# Initialize the PiCamera
picam2 = Picamera2()

# Configure the camera for preview
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

# Start the camera
picam2.start()

try:
    while True:
        # Capture a frame
        frame = picam2.capture_array()

        # Display the frame
        cv2.imshow("PiCamera3 Preview", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the camera and clean up
    picam2.stop()
    cv2.destroyAllWindows()
