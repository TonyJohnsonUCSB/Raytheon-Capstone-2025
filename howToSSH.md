# How to connect to SSH into a RPi from personal laptop

## Step 1: Find IP address on the Pi
* If using personal hotspot (recommended), IP will stay the same. Skip to next step if you already have it the IP from a previous hotspot connection
* If using Eduroam, the IP address will change every time 
    * Connect the PI to a monitor
    * Open a terminal in the PI and type "hostname -I" 
    * This is the IP address

## Step 2: Ensure personal laptop and Pi are on the same Wi-Fi network
## Step 3: Connect to SSH from personal laptop
* Open a terminal (command prompt, powershell, etc)
* Type "ssh rtxcapstone@123.456.789" (replacing that example IP addresss with the real one)
* Say yes if asked to save the SSH config file
* You should be connected from here, this is terminal on the pi which you can use to modify code and run commands. Repeat step 3 in another terminal if you need multiple (e.g. to run multiple scripts at a time)


