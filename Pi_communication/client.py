import socket
import time
import random  # Used if mode is random
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

# added global variable for coordinates
coordinates = ""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Client program to send GPS coordinates to a server.")
    parser.add_argument(
        "network", 
        choices=["eduroam", "hotspot"], 
        help="Specify the network type to connect (either 'eduroam' or 'hotspot')."
    )
    parser.add_argument(
        "mode",
        choices=["manual", "random"],
        help="Specify the mode of coordinate entry (either 'manual' or 'random')."
    )
    return parser.parse_args()

def get_gps_coordinates():
    """Generate random GPS coordinates."""
    latitude = round(random.uniform(-90.0, 90.0), 6)
    longitude = round(random.uniform(-180.0, 180.0), 6)
    return latitude, longitude

def get_user_coordinates():
    """Prompt the user for latitude and longitude, validate them, and return valid coords."""
    while True:
        try:
            lat_str = input("Enter latitude (between -90.0 and 90.0): ")
            lon_str = input("Enter longitude (between -180.0 and 180.0): ")

            # Convert to float
            latitude = float(lat_str)
            longitude = float(lon_str)

            # Validate ranges
            if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
                return latitude, longitude
            else:
                print("Invalid coordinates. Please ensure the values are within the specified ranges.")
        except ValueError:
            print("Invalid input. Please enter numeric values for latitude and longitude.")

def main():
    global coordinates  # Declare global to modify coordinates variable

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
            logging.error(f"Failed to connect to server: {e}")
            print(f"Failed to connect to server: {e}")
            return

        while True:
            if args.mode == "manual":
                # Get user-input coordinates
                latitude, longitude = get_user_coordinates()
            else:
                # Get random coordinates
                latitude, longitude = get_gps_coordinates()
            
            coordinates = f"{latitude},{longitude}"  # Update global variable
            
            logging.info(f"Sending GPS coordinates: {coordinates}")
            print(f"Sending GPS coordinates: {coordinates}")
            
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
            print("Waiting 5 seconds before sending the next coordinates...")
            time.sleep(5)

if __name__ == "__main__":
    main()
