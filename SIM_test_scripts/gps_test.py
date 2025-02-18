#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan
from geopy.distance import geodesic


async def run():
    """ Main function to connect, execute mission, and land """
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    async for position in drone.telemetry.position():
        home_lat = position.latitude_deg
        home_lon = position.longitude_deg
        print(f"Home GPS Coordinates: Lat {home_lat}, Lon {home_lon}")
        break

    print("-- Arming")
    await drone.action.arm()

    print("-- Creating mission")
    mission_plan = await create_mission(drone, home_lat, home_lon)

    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_plan)

    # Start mission progress monitoring
    print_mission_progress_task = asyncio.ensure_future(print_mission_progress(drone, home_lat, home_lon))
    running_tasks = [print_mission_progress_task]
    termination_task = asyncio.ensure_future(observe_is_in_air(drone, running_tasks))

    print("-- Starting mission")
    await drone.mission.start_mission()

    await termination_task


async def create_mission(drone, home_lat, home_lon):
    """ Creates a simple mission: take off, move 10 meters east, hold, land """
    altitude = 10  # Increased altitude for visibility
    speed = 2  # Increase speed to make movement noticeable
    hold_time = 10  # Increased hold time at waypoints

    # Calculate the new position 10 meters east
    target_lat, target_lon = get_new_position(home_lat, home_lon, 10, 0)
    print(f"Target GPS Coordinates (10m East): Lat {target_lat}, Lon {target_lon}")

    mission_items = [
        # Takeoff and hold for 10 seconds
        MissionItem(
            home_lat, home_lon, altitude, speed, True, hold_time, float('nan'),
            MissionItem.CameraAction.NONE, float('nan'), float('nan'),
            float('nan'), float('nan'), float('nan'),
            MissionItem.VehicleAction.NONE  # Explicit hold
        ),
        # Move 10 meters east and hold for 10 seconds
        MissionItem(
            target_lat, target_lon, altitude, speed, True, hold_time, float('nan'),
            MissionItem.CameraAction.NONE, float('nan'), float('nan'),
            float('nan'), float('nan'), float('nan'),
            MissionItem.VehicleAction.NONE  # Explicit hold
        ),
        # Land with an additional 10s pause
        MissionItem(
            target_lat, target_lon, 0, speed, True, hold_time, float('nan'),
            MissionItem.CameraAction.NONE, float('nan'), float('nan'),
            float('nan'), float('nan'), float('nan'),
            MissionItem.VehicleAction.LAND
        )
    ]

    return MissionPlan(mission_items)


def get_new_position(lat, lon, meters_east, meters_north):
    """ Calculate new GPS position using geopy """
    start_point = (lat, lon)

    # Move north/south
    new_point = geodesic(meters=meters_north).destination(start_point, 0)  # 0° is North
    new_lat = new_point.latitude

    # Move east/west
    new_point = geodesic(meters=meters_east).destination((new_lat, lon), 90)  # 90° is East
    new_lon = new_point.longitude

    return new_lat, new_lon


async def print_mission_progress(drone, home_lat, home_lon):
    """ Print mission progress as drone moves, showing relative change in lat/lon """
    async for mission_progress in drone.mission.mission_progress():
        async for position in drone.telemetry.position():
            lat_change = position.latitude_deg - home_lat
            lon_change = position.longitude_deg - home_lon

            print(f"Mission progress: {mission_progress.current}/{mission_progress.total} | "
                  f"ΔLat: {lat_change:.6e}, ΔLon: {lon_change:.6e}")
            break  # Only print once per mission progress update


async def observe_is_in_air(drone, running_tasks):
    """ Monitors whether the drone is in air and stops tasks after landing """
    was_in_air = False

    async for is_in_air in drone.telemetry.in_air():
        if is_in_air:
            was_in_air = True

        if was_in_air and not is_in_air:
            for task in running_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await asyncio.get_event_loop().shutdown_asyncgens()
            return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())
