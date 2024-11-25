from picamera2 import Picamera2
from libcamera import controls

# Initialize camera
picam2 = Picamera2()

# Create a basic configuration without preview
config = picam2.create_still_configuration()
picam2.configure(config)

# Start camera
picam2.start()

# Optional: Set some camera controls
picam2.set_controls({"ExposureTime": 1000, "AnalogueGain": 1.0})

# Capture an image
picam2.capture_file("test.jpg")

# Stop camera
picam2.stop()