from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.DRM)  # Ensure this matches your display method
picam2.configure(picam2.create_preview_configuration())

picam2.start()
time.sleep(5)  # Give time for the preview to be visible
picam2.stop()
