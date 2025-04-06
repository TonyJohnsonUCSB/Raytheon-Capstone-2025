# Drone Flying Procedure

## Before heading to Stroke Field...

- Check if LiPo and UPS batteries are charged
- (If using different hotspot) set up the hotspot and let the Pi connect to it

```shell
sudo su
ifdown wlan0
nano /etc/wpa_supplicant/wpa_supplicant.conf
(Change the corresponding "ssid" and "psk")
ifup wlan0
hostname -I
```

- Remember the first IP address (typically looks like 172.20.10.x for personal hotspot)
- Connect to the Pi on your laptop using RealVNC Viewer with the IP address
- Don't turn off the UPS and you are free to go!



## Remember to bring...

- The drone (of course)
- LiPo battery
- Pi connected to the hotspot and powered by UPS
- BACKUP BATTERIES!!!
- RTK base station connected to laptop
  - Radio module labeled "RTK UAV"
  - u-blox GPS module with cylindrical antenna



## Wi-Fi ssid and password

The new Wi-Fi in the Barn is eleg_capstone, password: jyg7aqiqkg.



## Charging LiPo battery

The setup for the 80W LiPo battery charger is already set so you typically don't need to change it.

![image-20250406131956281](C:\Users\Haoyan Duan\AppData\Roaming\Typora\typora-user-images\image-20250406131956281.png)

To charge the LiPo battery, make sure to connect both of these cables. On the charger, press "START/ENTER" and hold for about 3 seconds until you hear a sound and the screen shows "R: 6SER..." Press "START/ENTER" one more time and the battery will start charging. This charger will stop charging after the battery is full, and the "Li6S" section on the charger will alternate between displaying "Li6S" and "FULL".

To stop charging at any time, press "ESC/STOP". 



