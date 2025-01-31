import serial
import time

# Configure the serial port
serial_port = '/dev/ttyAMA0'  # Adjust this if using a different port
baud_rate = 57600

try:
    # Open the serial port
    ser = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=2)

    if ser.is_open:
        print(f"Serial port {serial_port} opened successfully at {baud_rate} baud.")

        # Send a single message
        test_message = "Hello, Pixhawk!\n"
        ser.write(test_message.encode('utf-8'))
        print(f"Sent: {test_message.strip()}")

        # Give some time to receive a response
        time.sleep(1)

        # Read the response (if any)
        response = ser.read(ser.in_waiting).decode('latin-1', errors='ignore')
        if response:
            print(f"Received: {response.strip()}")
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
