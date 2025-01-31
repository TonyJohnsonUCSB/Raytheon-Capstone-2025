from pymavlink import mavutil

# Connect to the Pixhawk's TELEM1 port
serial_port = '/dev/ttyAMA0'
baud_rate = 57600

try:
    # Establish MAVLink connection
    connection = mavutil.mavlink_connection(serial_port, baud=baud_rate)

    # Wait for a message (blocking mode)
    msg = connection.recv_match(blocking=True, timeout=5)

    if msg:
        # Print the received message as a dictionary
        print(f"Received MAVLink message: {msg.to_dict()}")
    else:
        print("No MAVLink message received.")

except Exception as e:
    print(f"Error: {e}")
finally:
    print("Connection closed.")
