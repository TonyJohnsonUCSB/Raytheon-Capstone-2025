import socket
import logging

# Setup logging
logging.basicConfig(
    filename="server_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define server parameters
HOST = ""  # Listen on all available interfaces
PORT = 65432

# Simulated verification function
def verify_coordinates(lat, lon):
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return True
    return False

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
