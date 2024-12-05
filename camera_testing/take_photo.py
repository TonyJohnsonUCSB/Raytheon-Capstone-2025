import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size":(1640,1232)},main={"format":'RGB888',"size":(640,480)}))
picam2.start()
time.sleep(2)

img = picam2.capture_array()

cv2.imshow("Outout",img)
cv2.waitKey(0)

picam2.stop()
picam2.close()
