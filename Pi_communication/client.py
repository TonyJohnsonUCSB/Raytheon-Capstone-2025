import socket
import time
import random  # Replace with GPS RTK module in production
import logging
import argparse

# Setup logging
logging.basicConfig(
    filename="client_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define server details
SERVER_IPS = {
    "hotspot": "192.168.4.1",
    "eduroam": "169.231.221.104"
}
PORT = 65432

# Simulate getting GPS coordinates (replace with GPS RTK module logic)
def get_gps_coordinates():
    latitude = round(random.uniform(-90.0, 90.0), 6)
    longitude = round(random.uniform(-180.0, 180.0), 6)
    return latitude, longitude

# Argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description="Client program to send GPS coordinates to a server.")
    parser.add_argument(
        "network", 
        choices=["eduroam", "hotspot"], 
        help="Specify the network type to connect (either 'eduroam' or 'hotspot')."
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Get the server IP based on the network argument
    SERVER_IP = SERVER_IPS.get(args.network)
    
    if not SERVER_IP:
        print("Invalid network type. Please choose either 'eduroam' or 'hotspot'.")
        return
    
    # Connect to the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            client_socket.connect((SERVER_IP, PORT))
            logging.info(f"Connected to server at {SERVER_IP}:{PORT}")
            print(f"Connected to server at {SERVER_IP}:{PORT}")
        except Exception as e:
<<<<<<< HEAD
            logging.error(f"Failed to connect to server: {e}")
            print(f"Failed to connect to server: {e}")
            return

        while True:
            # Get GPS-RTK coordinates
            latitude, longitude = get_gps_coordinates()
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

if __name__ == "__main__":
    main()
=======
            logging.error(f"Communication error: {e}")
            print(f"Communication error: {e}")
        
        # Simulate periodic transmission (every 5 seconds)
        time.sleep(5)
>>>>>>> 7c5fe954dbc3cd965964dfd8857b5afa2ff62c73
