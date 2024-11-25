import socket
import time
# import random  # Replace with GPS RTK module in production
import logging

# Setup logging
logging.basicConfig(
    filename="client_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define server details
SERVER_IP = "192.168.X.X"  # Replace with the receiving Raspberry Pi's IP
PORT = 65432

# Simulate getting GPS coordinates (replace with GPS RTK module logic)
#def get_gps_coordinates():
 #   latitude = round(random.uniform(-90.0, 90.0), 6)
  #  longitude = round(random.uniform(-180.0, 180.0), 6)
   # return latitude, longitude

# Example GPS coordinates for testing
def get_example_coordinates():
    # Replace these with any static or dynamic values for testing
    latitude = 37.7749  # Example: San Francisco latitude
    longitude = -122.4194  # Example: San Francisco longitude
    return latitude, longitude

# Connect to the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    try:
        client_socket.connect((SERVER_IP, PORT))
        logging.info(f"Connected to server at {SERVER_IP}:{PORT}")
        print(f"Connected to server at {SERVER_IP}:{PORT}")
    except Exception as e:
        logging.error(f"Failed to connect to server: {e}")
        print(f"Failed to connect to server: {e}")
        exit()

    while True:
        # Get GPS-RTK coordinates
        latitude, longitude = get_example_coordinates()
        coordinates = f"{latitude},{longitude}"
        logging.info(f"Sending GPS-RTK coordinates: {coordinates}")
        print(f"Sending GPS-RTK coordinates: {coordinates}")
        
        try:
            # Send coordinates to the server
            client_socket.sendall(coordinates.encode('utf-8'))
            # Wait for acknowledgment
            response = client_socket.recv(1024).decode('utf-8')
            logging.info(f"Server response: {response}")
            print(f"Server response: {response}")
        except Exception as e:
            logging.error(f"Communication error: {e}")
            print(f"Communication error: {e}")
        
        # Simulate periodic transmission (every 5 seconds)
        time.sleep(5)
