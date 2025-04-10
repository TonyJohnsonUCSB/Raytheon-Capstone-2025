import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD) #physical pin numbering

#initialize pins 
ENA = 32    
IN1 = 36    
IN2 = 38    

#setup GPIO pins
GPIO.setup(IN1, GPIO.OUT)  
GPIO.setup(IN2, GPIO.OUT)  
GPIO.setup(ENA, GPIO.OUT)

#set up PWM
pwm = GPIO.PWM(ENA, 10000)  #frequency of pwm set to 10000 Hz
pwm.start(0)  				#start pwm w 0% duty cycle
speed = 50 					#motor at half speed (50%)

#forward
def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

#backward
def move_backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)

#stop motor
def stop_motor():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)

#loop to test motor control
try:
    #enable H-Bridge
    GPIO.output(ENA, GPIO.HIGH)

    #forward for 5s
    print("Moving forward")
    move_forward()
    time.sleep(5)

    #backward for 5s
    print("Moving backward")
    move_backward()
    time.sleep(5)

    #stop motor
    print("Stopping")
    stop_motor()

finally:
    # Clean up GPIO settings before exiting
    GPIO.cleanup()
