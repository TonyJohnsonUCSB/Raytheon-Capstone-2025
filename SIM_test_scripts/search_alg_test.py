#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

def get_new_position(lat, lon, meters_east, meters_north):
    """
    Calculates a new latitude and longitude given a starting point, a movement in meters east and north.
    """
    earth_radius = 6378137.0  # Earth radius in meters
    new_lat = lat + (meters_north / earth_radius) * (180.0 / 3.141592653589793)
    new_lon = lon + (meters_east / (earth_radius * 
                   (3.141592653589793 / 180.0) * 
                   abs(lat)))
    return new_lat, new_lon

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    async for position in drone.telemetry.position():
        home_lat = position.latitude_deg
        home_lon = position.longitude_deg
        break

    mission_items = []

    altitude = 3  # 3 meters altitude
    speed = 1  # 1 m/s speed

    # Takeoff point
    mission_items.append(MissionItem(home_lat, home_lon, altitude, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.NONE))

    # Move 3 meters east
    lat, lon = get_new_position(home_lat, home_lon, 3, 0)
    mission_items.append(MissionItem(lat, lon, altitude, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.NONE))
    
    # Move 3 meters south
    lat, lon = get_new_position(lat, lon, 0, -3)
    mission_items.append(MissionItem(lat, lon, altitude, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.NONE))
    
    # Move 3 meters west
    lat, lon = get_new_position(lat, lon, -3, 0)
    mission_items.append(MissionItem(lat, lon, altitude, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.NONE))
    
    # Move 3 meters north (back to start position)
    lat, lon = get_new_position(lat, lon, 0, 3)
    mission_items.append(MissionItem(lat, lon, altitude, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.NONE))
    
    # Land
    mission_items.append(MissionItem(home_lat, home_lon, 0, speed, True,
                                     float('nan'), float('nan'), MissionItem.CameraAction.NONE,
                                     float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                     MissionItem.VehicleAction.LAND))

    mission_plan = MissionPlan(mission_items)

    await drone.mission.set_return_to_launch_after_mission(False)

    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_plan)

    print("-- Arming")
    await drone.action.arm()

    print("-- Starting mission")
    await drone.mission.start_mission()

    await monitor_mission_progress(drone)

async def monitor_mission_progress(drone):
    async for mission_progress in drone.mission.mission_progress():
        print(f"Mission progress: {mission_progress.current}/{mission_progress.total}")
        if mission_progress.current == mission_progress.total:
            print("-- Mission complete")
            break

if __name__ == "__main__":
    asyncio.run(run())
