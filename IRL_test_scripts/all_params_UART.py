#!/usr/bin/env python3

import asyncio
import csv
from mavsdk import System

async def run():
    # Connect to the drone
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:57600")

    # Get the list of parameters
    all_params = await drone.param.get_all_params()
    
    # Define the CSV file name
    csv_filename = "drone_parameters.csv"
    
    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter Name", "Value"])
        
        # Iterate through all int parameters
        for param in all_params.int_params:
            writer.writerow([param.name, param.value])
        
        # Iterate through all float parameters
        for param in all_params.float_params:
            writer.writerow([param.name, param.value])
    
    print(f"Parameters saved to {csv_filename}")

# Run the asyncio loop
asyncio.run(run())
