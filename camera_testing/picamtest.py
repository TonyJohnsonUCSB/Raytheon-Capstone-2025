import os
import time
import subprocess
from picamera2 import Picamera2

# Create directory if it doesn't exist
output_dir = "phototest"
os.makedirs(output_dir, exist_ok=True)

# Initialize the camera
picam = Picamera2()

# Configure the camera for video recording
picam.configure(picam.create_video_configuration())

# Start the camera
picam.start()

# Temporary H.264 output file
h264_path = os.path.join(output_dir, "video.h264")
mp4_path = os.path.join(output_dir, "video.mp4")

# Start recording
picam.start_and_record_video(h264_path)
print("Recording started...")

# Record for 6 seconds while printing "recording" each second
for i in range(6):
    print(f"Recording... {i + 1} second(s)")
    time.sleep(1)

# Stop recording
picam.stop_recording()

# Stop the camera
picam.stop()

print(f"Video saved to {h264_path}")

# Convert H.264 to MP4 using ffmpeg
print("Converting to MP4...")
subprocess.run([
    "ffmpeg", "-y", "-i", h264_path, "-c:v", "copy", "-f", "mp4", mp4_path
])

print(f"MP4 video saved to {mp4_path}")
