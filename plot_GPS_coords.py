import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Coordinates (lat, lon)
coordinates = [
    (34.4189167, -119.8553056),
    (34.4189722, -119.8553056),
    (34.4189722, -119.8551667),
    (34.4189167, -119.8551667)
]

# Convert to GeoDataFrame
points = [Point(lon, lat) for lat, lon in coordinates]
gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# Compute convex hull
hull = gdf.geometry.union_all().convex_hull
hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")

# Plot setup
fig, ax = plt.subplots()
gdf.plot(ax=ax, color='red', markersize=20)
hull_gdf.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)

ax.set_title("Geofence Convex Hull")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.axis("equal")
plt.show()
