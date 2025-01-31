import serial
import time

# Configure the serial port
serial_port = '/dev/ttyAMA0'  # Update if using a different port (e.g., '/dev/serial0')
from pymavlink import mavutil

# Open the connection to the Pixhawk's telemetry port
connection = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)

while True:
    # Wait for and receive a MAVLink message
    msg = connection.recv_match(blocking=True)
    if msg:
        print(f"Received MAVLink message: {msg.to_dict()}")
baud_rate = 57600

try:
    # Open the serial port
    ser = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=1)

    if ser.is_open:
        print(f"Serial port {serial_port} opened successfully at {baud_rate} baud.")

        # Send test data
        test_message = "Hello, Pixhawk!\n"
        ser.write(test_message.encode('utf-8'))
        print(f"Sent: {test_message.strip()}")

        # Wait for a response (if any)
        time.sleep(1)  # Adjust if necessary

        # Read response
        response = ser.read(ser.in_waiting or 1)
        if response:
            print(f"Received (raw bytes): {response.strip()}")
        else:
            print("No response received.")

    else:
        print("Failed to open serial port.")

except Exception as e:
    print(f"Error: {e}")

finally:
    # Close the serial port
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print(f"Serial port {serial_port} closed.")
