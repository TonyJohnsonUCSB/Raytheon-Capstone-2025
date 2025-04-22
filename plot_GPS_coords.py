import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx

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
hull = gdf.geometry.unary_union.convex_hull
hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")

# Reproject to Web Mercator for contextily
gdf_3857 = gdf.to_crs(epsg=3857)
hull_gdf_3857 = hull_gdf.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
gdf_3857.plot(ax=ax, color='red', markersize=20)
hull_gdf_3857.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

ax.set_title("Geofence Convex Hull on Satellite Map")
plt.axis("equal")
plt.show()
