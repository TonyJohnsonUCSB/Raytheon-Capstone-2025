#!/bin/bash

echo "Please select the network type:"
echo "1) eduroam"
echo "2) hotspot"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        network="eduroam"
        # Run the wifi_switch command to connect to eduroam
        echo "Switching to WiFi network 'eduroam'..."
        wifi_switch eduroam
        ;;
    2)
        network="hotspot"
        # You can optionally add a wifi_switch command for the hotspot network if needed
        echo "Switching to WiFi network 'hotspot'..."
        wifi_switch hotspot
        ;;
    *)
        echo "Invalid choice. Please select either 1 or 2."
        exit 1
        ;;
esac

echo "Running Python script with selected network type: $network"
sudo python3 client.py $network
