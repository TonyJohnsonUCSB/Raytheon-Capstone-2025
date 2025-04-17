from gpiozero import Servo
from gpiozero import AngularServo
from time import sleep
from gpiozero import LED
import time
import pigpio

pi = pigpio.pi()
servo_pin = 11
pi.set_servo_pulsewidth(servo_pin,1000)

# Create a servo object on GPIO pin 17.
# By default, gpiozero uses software PWM.
# servo = Servo(17)
# #signal = LED(17)

# t0 = time.time()
# #Angservo = AngularServo(17)

# print("Starting servo test...")
# while time.time()-t0< 60:    
    # print("Moving servo to maximum")
    # servo.value=1
    # sleep(5)

    # print("Moving servo to middle")
    # servo.value=0
    # sleep(5)

    # print("Moving servo to minimum")
    # servo.value=-1
    # sleep(5)
    
