from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.start_preview(alpha=192)
sleep(1)
camera.capture("/home/rtxcapstone/Documents/Raytheon-Capstone-2025")
camera.stop_preview()

