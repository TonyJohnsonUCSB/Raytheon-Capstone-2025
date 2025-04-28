import lgpio
import time

# Open a connection to the GPIO chip
h = lgpio.gpiochip_open(0)  # '0' is the default GPIO chip

# Initialize pins
ENA = 12   # Physical pin 32 is GPIO12
IN1 = 16   # Physical pin 36 is GPIO16
IN2 = 20   # Physical pin 38 is GPIO20

# Set modes for pins
lgpio.gpio_claim_output(h, IN1, 0)
lgpio.gpio_claim_output(h, IN2, 0)
lgpio.gpio_claim_output(h, ENA, 0)

# Setup PWM
freq = 10000  # 10 kHz
duty_cycle = 0
lgpio.tx_pwm(h, ENA, freq, duty_cycle)

speed = 100  # Motor speed in % duty cycle

def move_forward():
    lgpio.gpio_write(h, IN1, 1)
    lgpio.gpio_write(h, IN2, 0)
    lgpio.tx_pwm(h, ENA, freq, speed)

def move_backward():
    lgpio.gpio_write(h, IN1, 0)
    lgpio.gpio_write(h, IN2, 1)
    lgpio.tx_pwm(h, ENA, freq, speed)

def stop_motor():
    lgpio.gpio_write(h, IN1, 0)
    lgpio.gpio_write(h, IN2, 0)
    lgpio.tx_pwm(h, ENA, freq, 0)

try:
    # enable H-Bridge (already done by writing to ENA)
    print("Moving forward")
    move_forward()
    time.sleep(2)

    print("Moving backward")
    move_backward()
    time.sleep(2)

    print("Stopping")
    stop_motor()

finally:
    # Cleanup
    lgpio.gpiochip_close(h)
