from pymavlink import mavutil
import time

# === CONFIGURATION ===
LATITUDE = -34.3982      # Change these values each time you want a new waypoint
LONGITUDE = 149.5458
ALTITUDE = 10           # Meters above home

# === CONNECT TO SITL ===
print("[INFO] Connecting to SITL...")
master = mavutil.mavlink_connection('udp:172.24.208.1:14550')
master.wait_heartbeat()
print(f"[INFO] Connected to system {master.target_system}, component {master.target_component}")

# === HELPER: SET FLIGHT MODE ===
def set_mode(mode_name):
    mode_id = master.mode_mapping().get(mode_name.upper())
    if mode_id is None:
        raise ValueError(f"[ERROR] Unknown mode: {mode_name}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    print(f"[INFO] Mode set request: {mode_name}")
    time.sleep(1)

# === STEP 1: SWITCH TO HOLD TO STOP CURRENT MISSION ===
print("[INFO] Switching to HOLD mode...")
set_mode("HOLD")

# === STEP 2: CLEAR ANY EXISTING MISSION ===
print("[INFO] Clearing previous mission...")
master.mav.mission_clear_all_send(master.target_system, master.target_component)
time.sleep(1)

# === STEP 3: SEND NEW MISSION COUNT ===
print("[INFO] Uploading new waypoint...")
lat = int(LATITUDE * 1e7)
lon = int(LONGITUDE * 1e7)
master.mav.mission_count_send(master.target_system, master.target_component, 1)

# === STEP 4: WAIT FOR MISSION REQUEST ===
while True:
    msg = master.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'], blocking=True, timeout=5)
    if msg is None:
        print("[ERROR] Timeout waiting for MISSION_REQUEST.")
        exit(1)
    if msg.seq == 0:
        master.mav.mission_item_int_send(
            master.target_system,
            master.target_component,
            0,  # sequence
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0,  # current = 0 (won’t start until we say so)
            1,  # autocontinue = yes
            0, 0, 0, 0,
            lat,
            lon,
            ALTITUDE
        )
        print(f"[INFO] Sent waypoint: lat={LATITUDE}, lon={LONGITUDE}")
        break

# === STEP 5: WAIT FOR ACK ===
ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=5)
if ack:
    print(f"[INFO] Mission ACK: {ack.type}")
else:
    print("[WARNING] No mission ACK received.")

# === STEP 6: SET ACTIVE WP INDEX TO 0 ===
print("[INFO] Setting mission index to 0...")
master.mav.mission_set_current_send(
    master.target_system,
    master.target_component,
    0
)
time.sleep(1)

# === STEP 7: SWITCH TO AUTO MODE ===
print("[INFO] Switching to AUTO mode...")
set_mode("AUTO")

# === STEP 8: ARM ROVER ===
print("[INFO] Arming rover...")
master.arducopter_arm()
time.sleep(1)

# === STEP 9: START MISSION FROM WAYPOINT 0 ===
print("[INFO] Starting mission from waypoint 0...")
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_MISSION_START,
    0,
    0,  # start index
    0, 0, 0, 0, 0, 0
)

msg = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=5)
if msg and msg.command == mavutil.mavlink.MAV_CMD_MISSION_START:
    print(f"[INFO] Mission start ACK: {msg.result}")
else:
    print("[WARNING] No ACK for MISSION_START")

print("[✅] Done! Rover should now be heading toward the NEW waypoint.")
