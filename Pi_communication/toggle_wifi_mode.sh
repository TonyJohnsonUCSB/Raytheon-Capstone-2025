#!/bin/bash

# Function to switch to Eduroam
switch_to_eduroam() {
    echo "Switching to Eduroam..."
    # Stop hostapd and dnsmasq
    sudo systemctl stop hostapd
    sudo systemctl stop dnsmasq

    # Enable wpa_supplicant for wlan0
    sudo sed -i '/^nohook wpa_supplicant/d' /etc/dhcpcd.conf
    sudo systemctl restart dhcpcd
    sudo systemctl start wpa_supplicant

    # Restart the Wi-Fi interface
    sudo ifdown wlan0
    sudo ifup wlan0

    echo "Switched to Eduroam!"
}

# Function to switch to Hotspot
switch_to_hotspot() {
    echo "Switching to Hotspot..."
    # Stop wpa_supplicant
    sudo systemctl stop wpa_supplicant

    # Add nohook for wpa_supplicant in dhcpcd.conf
    if ! grep -q "nohook wpa_supplicant" /etc/dhcpcd.conf; then
        echo "nohook wpa_supplicant" | sudo tee -a /etc/dhcpcd.conf
    fi
    sudo systemctl restart dhcpcd

    # Set static IP for wlan0
    sudo ifconfig wlan0 192.168.4.1

    # Start hostapd and dnsmasq
    sudo systemctl start hostapd
    sudo systemctl start dnsmasq

    echo "Hotspot is now active!"
}

# Main menu
echo "Select mode:"
echo "1) Eduroam"
echo "2) Hotspot"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" -eq 1 ]; then
    switch_to_eduroam
elif [ "$choice" -eq 2 ]; then
    switch_to_hotspot
else
    echo "Invalid choice. Exiting."
    exit 1
fi
