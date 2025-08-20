"""
Static Satellite Map PNG Generator
=================================

Creates a high-quality PNG image with satellite imagery background
showing the SESRO test locations and analysis grid.

Uses matplotlib + contextily for static satellite imagery overlay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import sys
import os
import math
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for coordinate service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coordinate_service import get_coordinate_service


def get_grid_indices(x_coord, y_coord, min_x, min_y, quadrant_size):
    """Map coordinates to grid indices"""
    col_idx = int((x_coord - min_x) // quadrant_size)
    row_idx = int((y_coord - min_y) // quadrant_size)
    return row_idx, col_idx


def create_occupancy_matrix(x_coords, y_coords, quadrant_size=250):
    """Create systematic grid matrix to track quadrant occupancy with 1s and 0s"""
    log_message("Creating systematic grid matrix for quadrant analysis...")

    # Define site boundaries with buffer
    buffer = 500  # 500m buffer
    min_x_site = min(x_coords) - buffer
    max_x_site = max(x_coords) + buffer
    min_y_site = min(y_coords) - buffer
    max_y_site = max(y_coords) + buffer

    # Calculate grid dimensions
    site_width = max_x_site - min_x_site
    site_height = max_y_site - min_y_site

    num_cols = int(np.ceil(site_width / quadrant_size))
    num_rows = int(np.ceil(site_height / quadrant_size))

    log_message(
        f"Grid dimensions: {num_rows} rows x {num_cols} cols ({num_rows * num_cols} total quadrants)"
    )

    # Initialize grid matrix with zeros (empty quadrants)
    grid_matrix = np.zeros((num_rows, num_cols), dtype=int)

    # Map test locations to grid and mark occupied quadrants
    occupied_count = 0
    for x, y in zip(x_coords, y_coords):
        row, col = get_grid_indices(x, y, min_x_site, min_y_site, quadrant_size)

        # Ensure indices are within bounds
        if 0 <= row < num_rows and 0 <= col < num_cols:
            if grid_matrix[row, col] == 0:  # First time occupying this quadrant
                occupied_count += 1
            grid_matrix[row, col] = 1
        else:
            log_message(
                f"Warning: Test location ({x:.0f}, {y:.0f}) outside grid boundaries",
                "WARNING",
            )

    log_message(
        f"Grid analysis complete: {occupied_count} occupied, {(num_rows * num_cols) - occupied_count} empty quadrants"
    )

    return (
        grid_matrix,
        min_x_site,
        min_y_site,
        max_x_site,
        max_y_site,
        num_rows,
        num_cols,
    )


def render_grid_and_quadrants(
    ax, grid_matrix, min_x_site, min_y_site, num_rows, num_cols, quadrant_size=250
):
    """Render grid lines and quadrant highlighting based on occupancy matrix"""
    log_message("Rendering systematic grid and quadrant highlighting...")

    # Draw grid lines
    for r in range(num_rows + 1):
        y_line = min_y_site + r * quadrant_size
        ax.axhline(y_line, color="white", alpha=0.4, linewidth=0.5, zorder=3)

    for c in range(num_cols + 1):
        x_line = min_x_site + c * quadrant_size
        ax.axvline(x_line, color="white", alpha=0.4, linewidth=0.5, zorder=3)

    # Highlight quadrants based on occupancy matrix
    occupied_quadrants = 0
    empty_quadrants = 0

    for r in range(num_rows):
        for c in range(num_cols):
            # Calculate quadrant coordinates
            quad_x = min_x_site + c * quadrant_size
            quad_y = min_y_site + r * quadrant_size

            if grid_matrix[r, c] == 1:
                # Occupied quadrant - #003780 with higher alpha
                color = "#003780"
                alpha = 0.15
                occupied_quadrants += 1
            else:
                # Empty quadrant - orange with lower alpha
                color = "orange"
                alpha = 0.00  # Very low alpha for dataless squares as requested
                empty_quadrants += 1

            # Add quadrant rectangle
            rect = plt.Rectangle(
                (quad_x, quad_y),
                quadrant_size,
                quadrant_size,
                facecolor=color,
                alpha=alpha,
                edgecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

    log_message(
        f"Rendered {occupied_quadrants} data quadrants (blue) + {empty_quadrants} empty quadrants (hidden)"
    )


def log_message(message: str, level: str = "INFO"):
    """Log processing steps with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def load_optimized_data():
    """Load the optimized Cu intersection data"""
    log_message("Loading optimized Cu intersection data...")

    try:
        # Try to load the optimized data first
        df = pd.read_csv("outputs/cu_intersections_optimized.csv")
        log_message(f"Loaded optimized data: {len(df)} locations")
        return df
    except FileNotFoundError:
        # Fallback to creating from cleaned data
        log_message("Optimized data not found, loading from cleaned dataset...")
        df = pd.read_csv("outputs/CompiledCu_Cleaned.csv")

        # Get unique locations with basic aggregation
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

        # Add coordinate transformation
        coord_service = get_coordinate_service()
        lats, lons = coord_service.bng_to_wgs84_vectorized(
            unique_locations["Easting"].values, unique_locations["Northing"].values
        )
        unique_locations["Latitude"] = lats
        unique_locations["Longitude"] = lons

        log_message(f"Processed fallback data: {len(unique_locations)} locations")
        return unique_locations


def create_satellite_png():
    """Create static PNG with satellite imagery using contextily"""
    log_message("Creating static satellite map PNG...")

    # Load data
    df = load_optimized_data()

    # Set up the plot with proper projection (Web Mercator for satellite tiles)
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)

    # Convert coordinates to Web Mercator for contextily
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_coords, y_coords = transformer.transform(
        df["Longitude"].values, df["Latitude"].values
    )

    # Create systematic grid matrix for quadrant analysis
    grid_matrix, min_x_site, min_y_site, max_x_site, max_y_site, num_rows, num_cols = (
        create_occupancy_matrix(x_coords, y_coords, quadrant_size=250)
    )

    # Prepare test type coloring
    test_types = (
        df["PrimaryTestType"]
        if "PrimaryTestType" in df.columns
        else df.get("TestTypes", df.get("TestType", ["Unknown"] * len(df)))
    )

    # Create color mapping for test types
    unique_test_types = set()
    for test_type_str in test_types:
        if isinstance(test_type_str, str):
            unique_test_types.update([t.strip() for t in test_type_str.split(",")])
        else:
            unique_test_types.add(str(test_type_str))

    unique_test_types = sorted(unique_test_types)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_test_types)))
    color_map = dict(zip(unique_test_types, colors))

    log_message(f"Found test types: {unique_test_types}")

    # Assign colors to each point (use first test type if multiple)
    point_colors = []
    for test_type_str in test_types:
        if isinstance(test_type_str, str) and "," in test_type_str:
            first_type = test_type_str.split(",")[0].strip()
        else:
            first_type = str(test_type_str).strip()
        point_colors.append(color_map.get(first_type, color_map.get("Unknown", "gray")))

    # Create the scatter plot colored by test type
    ax.scatter(
        x_coords,
        y_coords,
        c=point_colors,
        s=80,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )

    # Set the extent for the map using grid boundaries
    ax.set_xlim(min_x_site, max_x_site)
    ax.set_ylim(min_y_site, max_y_site)

    # Add satellite imagery using contextily
    log_message("Downloading satellite imagery tiles...")
    try:
        # Try satellite imagery first with reduced alpha
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto",
            alpha=0.5,  # Reduced alpha as requested
        )
        log_message("Satellite imagery added successfully")
    except Exception as e:
        log_message(f"Satellite imagery failed, using OpenStreetMap: {e}", "WARNING")
        try:
            # Fallback to OpenStreetMap
            ctx.add_basemap(
                ax,
                crs="EPSG:3857",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom="auto",
                alpha=0.5,  # Reduced alpha as requested
            )
            log_message("OpenStreetMap basemap added as fallback")
        except Exception as e2:
            log_message(f"All basemap attempts failed: {e2}", "ERROR")
            # Continue without basemap

    # Render systematic grid and quadrant highlighting
    render_grid_and_quadrants(
        ax, grid_matrix, min_x_site, min_y_site, num_rows, num_cols, quadrant_size=250
    )

    # Add site boundary
    site_x = [min(x_coords), max(x_coords), max(x_coords), min(x_coords), min(x_coords)]
    site_y = [min(y_coords), min(y_coords), max(y_coords), max(y_coords), min(y_coords)]
    ax.plot(
        site_x,
        site_y,
        color="yellow",
        linewidth=3,
        alpha=0.9,
        zorder=6,
        label="Site Boundary",
    )

    # Create legend for test types
    legend_elements = []
    for test_type, color in color_map.items():
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=test_type,
                markeredgecolor="black",
            )
        )

    # Add quadrant highlighting legend
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#003780",
            markersize=10,
            alpha=0.3,
            label="Quadrants with Data",
        )
    )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            alpha=0.3,
            label="Quadrants without Data",
        )
    )
    legend_elements.append(
        plt.Line2D([0], [0], color="yellow", linewidth=3, label="Site Boundary")
    )

    # Styling
    ax.set_title(
        "SESRO Site: Test Locations by Type on Satellite Imagery\n"
        + f"{len(df)} Test Locations with Quadrant Analysis (250m grid)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend with custom elements
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    # Add north arrow (simple)
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.annotate(
        "↑",
        xy=(0.95, 0.92),
        xycoords="axes fraction",
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
    )

    # Add scale information
    scale_text = f"Site Area: ~{((max(df['Easting']) - min(df['Easting'])) * (max(df['Northing']) - min(df['Northing']))) / 1e6:.1f} km²"
    ax.text(
        0.02,
        0.02,
        scale_text,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # Save the plot
    plt.tight_layout()

    output_file = "outputs/satellite_map_static.png"
    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )

    log_message(f"Static satellite map saved: {output_file}", "SUCCESS")

    # Show the plot
    plt.show()

    return output_file


def main():
    """Main execution function"""
    log_message("=== STATIC SATELLITE MAP GENERATOR START ===")

    try:
        output_file = create_satellite_png()

        print(f"\n{'='*60}")
        print("STATIC SATELLITE MAP COMPLETE")
        print(f"{'='*60}")
        print(f"📁 Output file: {output_file}")
        print("🛰️ High-resolution PNG with satellite imagery")
        print("📍 All test locations plotted by type")
        print("🔲 250m quadrant analysis overlay")
        print("📊 Color-coded by test type")
        print(f"{'='*60}")

    except Exception as e:
        log_message(f"Static map generation failed: {str(e)}", "ERROR")
        raise


if __name__ == "__main__":
    main()
