#!/bin/bash

# Function to switch to Eduroam
switch_to_eduroam() {
    echo "Switching to Eduroam..."
    # Stop hostapd and dnsmasq services
    sudo systemctl stop hostapd
    sudo systemctl stop dnsmasq

    # Enable wpa_supplicant for wlan0
    sudo sed -i '/^nohook wpa_supplicant/d' /etc/dhcpcd.conf
    sudo systemctl restart dhcpcd
    sudo systemctl start wpa_supplicant

    # Restart the Wi-Fi interface to connect to Eduroam
    sudo ifdown wlan0 || true
    sudo ifup wlan0 || true

    echo "Switched to Eduroam!"
}

# Function to switch to Hotspot
switch_to_hotspot() {
    echo "Switching to Hotspot..."
    # Stop wpa_supplicant service
    sudo systemctl stop wpa_supplicant

    # Add "nohook" for wpa_supplicant in dhcpcd.conf to disable it for wlan0
    if ! grep -q "nohook wpa_supplicant" /etc/dhcpcd.conf; then
        echo "nohook wpa_supplicant" | sudo tee -a /etc/dhcpcd.conf
    fi
    sudo systemctl restart dhcpcd

    # Assign a static IP to wlan0 for the hotspot
    sudo ifconfig wlan0 192.168.4.1 netmask 255.255.255.0

    # Start hostapd and dnsmasq services for the hotspot
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
