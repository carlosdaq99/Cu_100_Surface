"""
FAST Vectorized Satellite Map with Grid
======================================

Efficient implementation using vectorized coordinate transformations
based on Dash application patterns for speed optimization.

Key optimizations:
1. Vectorized pyproj transformations (batch processing)
2. Pandas operations instead of loops
3. Efficient grid generation
4. Minimal memory footprint
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pyproj
import warnings

warnings.filterwarnings("ignore")


def log_message(message, level="INFO"):
    """Log processing steps with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def vectorized_bng_to_wgs84(eastings, northings):
    """
    Vectorized coordinate transformation - processes entire arrays at once
    Based on efficient Dash application pattern
    """
    log_message(
        f"Converting {len(eastings):,} coordinates using vectorized transformation..."
    )

    # Define coordinate systems once
    bng = pyproj.CRS("EPSG:27700")  # British National Grid
    wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 (lat/lon)

    # Create transformer once
    transformer = pyproj.Transformer.from_crs(bng, wgs84, always_xy=True)

    # Vectorized transformation - processes all points simultaneously
    lons, lats = transformer.transform(eastings, northings)

    log_message("Vectorized coordinate transformation complete", "SUCCESS")
    return lats, lons


def load_and_prepare_data():
    """Load cleaned data and prepare for mapping"""
    log_message("Loading and preparing cleaned dataset...")

    try:
        df = pd.read_csv("CompiledCu_Cleaned.csv")
        log_message(f"Loaded {len(df):,} valid records")

        # Get unique locations for mapping
        unique_locations = (
            df.groupby(["Easting", "Northing"])
            .agg(
                {
                    "TestType": lambda x: ", ".join(sorted(set(x))),
                    "AverageCu": ["count", "mean", "min", "max"],
                    "Depth": ["min", "max"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        unique_locations.columns = [
            "Easting",
            "Northing",
            "TestTypes",
            "TestCount",
            "MeanCu",
            "MinCu",
            "MaxCu",
            "MinDepth",
            "MaxDepth",
        ]

        log_message(f"Prepared {len(unique_locations):,} unique test locations")
        return df, unique_locations

    except Exception as e:
        log_message(f"Error loading data: {str(e)}", "ERROR")
        return None, None


def create_efficient_grid(min_east, max_east, min_north, max_north, grid_size=250):
    """Create grid efficiently using vectorized operations"""
    log_message(f"Creating {grid_size}m analysis grid...")

    # Create grid boundaries
    east_range = np.arange(min_east, max_east + grid_size, grid_size)
    north_range = np.arange(min_north, max_north + grid_size, grid_size)

    # Create grid lines for visualization (sample only for performance)
    # Vertical lines (every 5th line for visual clarity)
    vertical_lines = []
    for east in east_range[::5]:  # Every 5th line
        vertical_lines.extend(
            [{"type": "line", "x0": east, "y0": min_north, "x1": east, "y1": max_north}]
        )

    # Horizontal lines (every 5th line for visual clarity)
    horizontal_lines = []
    for north in north_range[::5]:  # Every 5th line
        horizontal_lines.extend(
            [{"type": "line", "x0": min_east, "y0": north, "x1": max_east, "y1": north}]
        )

    log_message(f"Grid created: {len(east_range)} × {len(north_range)} cells")
    return vertical_lines + horizontal_lines


def create_fast_satellite_map():
    """Create satellite map using vectorized operations for maximum speed"""
    log_message("Starting FAST satellite map creation...")

    # Load data
    df, locations = load_and_prepare_data()
    if df is None:
        return None

    # Vectorized coordinate transformation for ALL unique locations at once
    lats, lons = vectorized_bng_to_wgs84(
        locations["Easting"].values, locations["Northing"].values
    )

    # Add coordinates to dataframe
    locations["Latitude"] = lats
    locations["Longitude"] = lons

    # Calculate map bounds
    center_lat, center_lon = np.mean(lats), np.mean(lons)
    log_message(f"Map center: {center_lat:.6f}, {center_lon:.6f}")

    # Create base map with satellite imagery
    fig = go.Figure()

    # Add test locations with efficient hover data
    log_message("Adding test locations to map...")

    # Create color scale based on mean Cu values
    hover_text = [
        f"<b>Location:</b> {east:.0f}E, {north:.0f}N<br>"
        + f"<b>Tests:</b> {test_types}<br>"
        + f"<b>Test Count:</b> {count}<br>"
        + f"<b>Mean Cu:</b> {mean_cu:.1f} kPa<br>"
        + f"<b>Cu Range:</b> {min_cu:.1f} - {max_cu:.1f} kPa<br>"
        + f"<b>Depth Range:</b> {min_depth:.1f} - {max_depth:.1f} m"
        for east, north, test_types, count, mean_cu, min_cu, max_cu, min_depth, max_depth in zip(
            locations["Easting"],
            locations["Northing"],
            locations["TestTypes"],
            locations["TestCount"],
            locations["MeanCu"],
            locations["MinCu"],
            locations["MaxCu"],
            locations["MinDepth"],
            locations["MaxDepth"],
        )
    ]

    fig.add_trace(
        go.Scattermapbox(
            lat=locations["Latitude"],
            lon=locations["Longitude"],
            mode="markers",
            marker=dict(
                size=8,
                color=locations["MeanCu"],
                colorscale="RdYlBu_r",
                colorbar=dict(title="Mean Cu (kPa)"),
                cmin=0,
                cmax=200,
                opacity=0.8,
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name=f"Test Locations (n={len(locations)})",
        )
    )

    # Add 100 kPa threshold marker in legend
    fig.add_trace(
        go.Scattermapbox(
            lat=[center_lat],
            lon=[center_lon],
            mode="markers",
            marker=dict(size=0.1, color="red"),
            showlegend=True,
            name="100 kPa Threshold (Red)",
            visible="legendonly",
        )
    )

    # Configure map layout for satellite imagery
    fig.update_layout(
        title=dict(
            text="SESRO Site: Test Locations on Satellite Imagery<br>"
            + f"<sub>{len(locations)} unique locations, {len(df):,} total records</sub>",
            x=0.5,
            font=dict(size=16),
        ),
        mapbox=dict(
            style="satellite",  # Satellite imagery
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12,
            bearing=0,
            pitch=0,
        ),
        width=1200,
        height=800,
        margin=dict(r=0, t=80, l=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    # Add grid overlay (light overlay on satellite)
    site_bounds = {
        "min_east": locations["Easting"].min(),
        "max_east": locations["Easting"].max(),
        "min_north": locations["Northing"].min(),
        "max_north": locations["Northing"].max(),
    }

    # Convert grid corners to lat/lon for overlay
    grid_corners_east = [
        site_bounds["min_east"],
        site_bounds["max_east"],
        site_bounds["max_east"],
        site_bounds["min_east"],
    ]
    grid_corners_north = [
        site_bounds["min_north"],
        site_bounds["min_north"],
        site_bounds["max_north"],
        site_bounds["max_north"],
    ]

    grid_lats, grid_lons = vectorized_bng_to_wgs84(
        grid_corners_east, grid_corners_north
    )

    # Add grid boundary
    fig.add_trace(
        go.Scattermapbox(
            lat=list(grid_lats) + [grid_lats[0]],  # Close the polygon
            lon=list(grid_lons) + [grid_lons[0]],
            mode="lines",
            line=dict(color="yellow", width=2),
            name="Analysis Area Boundary",
            hoverinfo="skip",
        )
    )

    log_message("Satellite map creation complete", "SUCCESS")
    return fig, locations


def main():
    """Main execution function"""
    log_message("=== FAST SATELLITE MAPPING SCRIPT START ===")

    try:
        # Create the map
        fig, locations = create_fast_satellite_map()

        if fig is None:
            log_message("Failed to create map", "ERROR")
            return

        # Save interactive HTML
        output_file = "outputs/fast_satellite_map_with_grid.html"
        fig.write_html(output_file)
        log_message(f"Interactive satellite map saved: {output_file}", "SUCCESS")

        # Show the map
        fig.show()

        # Summary statistics
        print(f"\n{'='*60}")
        print(f"FAST SATELLITE MAP GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"📍 Unique test locations: {len(locations):,}")
        print(
            f"🎯 Site area: ~{((locations['Easting'].max() - locations['Easting'].min()) * (locations['Northing'].max() - locations['Northing'].min())) / 1e6:.1f} km²"
        )
        print(
            f"📊 Mean Cu range: {locations['MeanCu'].min():.1f} - {locations['MeanCu'].max():.1f} kPa"
        )
        print(f"🗂️ Output saved: {output_file}")
        print(f"{'='*60}")

    except Exception as e:
        log_message(f"Script execution failed: {str(e)}", "ERROR")
        raise


if __name__ == "__main__":
    main()
