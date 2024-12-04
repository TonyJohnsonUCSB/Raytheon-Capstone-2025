from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
import time

def preview_video():
    # Initialize PiCamera2
    picam2 = Picamera2()

    # Configure the camera for preview
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)

    try:
        # Start the preview
        print("Starting preview...")
        picam2.start_preview()
        picam2.start()

        # Keep the preview on for 5 seconds
        time.sleep(5)

    finally:
        # Stop the preview
        print("Stopping preview...")
        picam2.stop_preview()
        picam2.close()

if __name__ == "__main__":
    # Ensure the script is running in a GUI or on a connected HDMI display
    preview_video()
