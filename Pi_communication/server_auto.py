import socket
import logging
import threading
import subprocess
import time

# Setup logging
logging.basicConfig(
    filename="server_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define server parameters
HOST = ""  # listen all hosts
PORT = 65432

# Simulated verification function
def verify_coordinates(lat, lon):
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return True
    return False

# Function to check and restart hotspot if needed
def monitor_hotspot(interval=30):
    def check_hotspot_status():
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "hostapd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout.strip() == "active"
        except Exception as e:
            logging.error(f"Error checking hotspot status: {e}")
            return False

    def start_hotspot():
        try:
            subprocess.run(["sudo", "systemctl", "start", "hostapd"], check=True)
            logging.info("Hotspot started successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start hotspot: {e}")
        except Exception as e:
            logging.error(f"Error starting hotspot: {e}")

    while True:
        if not check_hotspot_status():
            logging.warning("Hotspot is OFF. Restarting...")
            start_hotspot()
        time.sleep(interval)

# Start the hotspot monitoring in a separate thread
hotspot_thread = threading.Thread(target=monitor_hotspot, args=(30,), daemon=True)
hotspot_thread.start()

# Start server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        logging.info(f"Server listening on port {PORT}")
        print(f"Server listening on port {PORT}")
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        print(f"Failed to start server: {e}")
        exit()

    conn, addr = server_socket.accept()
    logging.info(f"Connected by {addr}")
    print(f"Connected by {addr}")

    with conn:
        while True:
            try:
                # Receive data
                data = conn.recv(1024)
                if not data:
                    break

                coordinates = data.decode('utf-8')
                logging.info(f"Received GPS-RTK coordinates: {coordinates}")
                print(f"Received GPS-RTK coordinates: {coordinates}")

                try:
                    latitude, longitude = map(float, coordinates.split(","))
                except ValueError:
                    response = "Invalid coordinates format"
                    conn.sendall(response.encode('utf-8'))
                    logging.warning(f"Sent response: {response}")
                    continue

                # Verify coordinates
                if verify_coordinates(latitude, longitude):
                    response = "Coordinates are correct"
                else:
                    response = "Coordinates are incorrect"

                # Send acknowledgment back to client
                conn.sendall(response.encode('utf-8'))
                logging.info(f"Sent response: {response}")
            except Exception as e:
                logging.error(f"Error during communication: {e}")
                print(f"Error during communication: {e}")
