#!/usr/bin/env python3
"""
Enhanced Test Location Mapping for SESRO 3D Surface Modeling
Phase 1.1: Test Location Analysis with 250m Grid and Satellite Imagery

This script maps the spatial distribution of geotechnical test locations across the SESRO site,
adds a 250m analysis grid, and integrates satellite imagery for enhanced visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy.spatial.distance import cdist
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
from pyproj import Proj, transform

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


def load_and_process_data(csv_path):
    """Load and process geotechnical data for location analysis"""
    print("📊 Loading geotechnical data...")

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} geotechnical records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

    # Data validation
    required_cols = ["Easting", "Northing", "TestType", "Depth_m", "AverageCu_kPa"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None, None

    # Clean data
    df_clean = df.dropna(subset=["Easting", "Northing", "TestType"]).copy()
    print(f"✅ After cleaning: {len(df_clean):,} records")

    # Group by location to get unique test sites
    print("\n🎯 Processing unique test locations...")

    # Round coordinates to nearest meter for grouping
    df_clean["Easting_rounded"] = df_clean["Easting"].round(0)
    df_clean["Northing_rounded"] = df_clean["Northing"].round(0)

    # Aggregate data by location
    location_groups = (
        df_clean.groupby(["Easting_rounded", "Northing_rounded"])
        .agg(
            {
                "TestType": lambda x: "/".join(sorted(x.unique())),
                "Depth_m": ["min", "max", "count"],
                "AverageCu_kPa": ["min", "max", "mean", "count"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    location_groups.columns = [
        "Easting",
        "Northing",
        "TestTypes",
        "MinDepth",
        "MaxDepth",
        "DepthRecords",
        "MinCu",
        "MaxCu",
        "MeanCu",
        "CuRecords",
    ]

    print(f"✅ Identified {len(location_groups)} unique test locations")

    return df_clean, location_groups


def analyze_spatial_coverage(locations):
    """Analyze spatial coverage and distribution of test locations"""
    print("\n📏 SPATIAL COVERAGE ANALYSIS:")

    # Site boundaries
    bounds = {
        "easting_min": locations["Easting"].min(),
        "easting_max": locations["Easting"].max(),
        "northing_min": locations["Northing"].min(),
        "northing_max": locations["Northing"].max(),
    }

    # Site dimensions
    site_width = bounds["easting_max"] - bounds["easting_min"]
    site_height = bounds["northing_max"] - bounds["northing_min"]
    total_area = (site_width * site_height) / 1_000_000  # km²

    print(f"Site boundaries (BNG coordinates):")
    print(f"  Easting: {bounds['easting_min']:.0f} to {bounds['easting_max']:.0f} m")
    print(f"  Northing: {bounds['northing_min']:.0f} to {bounds['northing_max']:.0f} m")
    print(f"Site dimensions: {site_width/1000:.1f} km × {site_height/1000:.1f} km")
    print(f"Total area: {total_area:.1f} km²")

    # Nearest neighbor analysis
    coords = locations[["Easting", "Northing"]].values
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)  # Exclude self-distances
    nearest_distances = np.min(distances, axis=1)

    print(f"\nNearest neighbor distances:")
    print(f"  Mean: {np.mean(nearest_distances):.0f} m")
    print(f"  Median: {np.median(nearest_distances):.0f} m")
    print(f"  Std: {np.std(nearest_distances):.0f} m")
    print(f"  Min: {np.min(nearest_distances):.0f} m")
    print(f"  Max: {np.max(nearest_distances):.0f} m")

    # Test density
    test_density = len(locations) / total_area
    print(f"\nTest density: {test_density:.1f} locations/km²")

    coverage_stats = {
        "bounds": bounds,
        "site_width_km": site_width / 1000,
        "site_height_km": site_height / 1000,
        "total_area_km2": total_area,
        "mean_nn_distance": np.mean(nearest_distances),
        "median_nn_distance": np.median(nearest_distances),
        "std_nn_distance": np.std(nearest_distances),
        "test_density": test_density,
    }

    return coverage_stats


def generate_analysis_grid(locations, grid_spacing=250):
    """Generate 250m analysis grid for spatial interpolation"""
    print(f"\n🔲 Generating {grid_spacing}m analysis grid...")

    # Get site bounds with buffer
    buffer = grid_spacing
    easting_min = locations["Easting"].min() - buffer
    easting_max = locations["Easting"].max() + buffer
    northing_min = locations["Northing"].min() - buffer
    northing_max = locations["Northing"].max() + buffer

    # Generate grid coordinates
    x_coords = np.arange(easting_min, easting_max + grid_spacing, grid_spacing)
    y_coords = np.arange(northing_min, northing_max + grid_spacing, grid_spacing)

    print(f"Grid dimensions: {len(x_coords)} × {len(y_coords)} cells")
    print(f"Total grid cells: {len(x_coords) * len(y_coords):,}")

    # Create grid cell centers
    grid_centers = []
    for x in x_coords[:-1]:  # Exclude last point to avoid edge cells
        for y in y_coords[:-1]:
            grid_centers.append(
                {
                    "CenterX": x + grid_spacing / 2,
                    "CenterY": y + grid_spacing / 2,
                    "MinX": x,
                    "MaxX": x + grid_spacing,
                    "MinY": y,
                    "MaxY": y + grid_spacing,
                }
            )

    grid_cells = pd.DataFrame(grid_centers)
    print(f"✅ Created {len(grid_cells):,} grid cells")

    return grid_cells, x_coords, y_coords


def assign_tests_to_grid(locations, grid_cells):
    """Assign test locations to grid cells for analysis"""
    print("\n📍 Assigning test locations to grid cells...")

    # Initialize test count column
    grid_cells["TestCount"] = 0
    grid_cells["TestTypes"] = ""

    # Assign each test location to appropriate grid cell
    for _, location in locations.iterrows():
        x, y = location["Easting"], location["Northing"]

        # Find containing grid cell
        mask = (
            (grid_cells["MinX"] <= x)
            & (x < grid_cells["MaxX"])
            & (grid_cells["MinY"] <= y)
            & (y < grid_cells["MaxY"])
        )

        if mask.any():
            idx = grid_cells[mask].index[0]
            grid_cells.loc[idx, "TestCount"] += 1

            # Track test types in cell
            current_types = grid_cells.loc[idx, "TestTypes"]
            if current_types:
                combined_types = current_types + "/" + location["TestTypes"]
            else:
                combined_types = location["TestTypes"]
            grid_cells.loc[idx, "TestTypes"] = combined_types

    # Summary statistics
    cells_with_tests = grid_cells[grid_cells["TestCount"] > 0]
    print(f"Grid cells with test locations: {len(cells_with_tests)}")
    print(f"Grid cells without tests: {len(grid_cells) - len(cells_with_tests)}")

    if len(cells_with_tests) > 0:
        print(f"Tests per cell statistics:")
        print(f"  Mean: {cells_with_tests['TestCount'].mean():.1f}")
        print(f"  Max: {cells_with_tests['TestCount'].max()}")
        print(
            f"  Cells with >1 test: {len(cells_with_tests[cells_with_tests['TestCount'] > 1])}"
        )

    return grid_cells


def analyze_test_types(locations):
    """Analyze distribution of test types across locations"""
    print("\n🔬 TEST TYPE ANALYSIS:")

    # Count locations by primary test type
    primary_test_types = locations["TestTypes"].str.split("/").str[0]
    test_counts = primary_test_types.value_counts()

    print("Locations by primary test type:")
    for test_type, count in test_counts.items():
        percentage = (count / len(locations)) * 100
        print(f"  {test_type}: {count} locations ({percentage:.1f}%)")

    # Analyze mixed test type locations
    mixed_locations = locations[locations["TestTypes"].str.contains("/")]
    print(
        f"\nLocations with multiple test types: {len(mixed_locations)} ({len(mixed_locations)/len(locations)*100:.1f}%)"
    )

    if len(mixed_locations) > 0:
        print("Mixed test type combinations:")
        mixed_counts = mixed_locations["TestTypes"].value_counts()
        for combo, count in mixed_counts.head(5).items():
            print(f"  {combo}: {count} locations")

    return test_counts


def create_static_map_with_grid(
    locations, grid_cells, x_coords, y_coords, coverage_stats, output_dir
):
    """Create enhanced static map with grid overlay and satellite imagery"""
    print("\n🗺️ Creating enhanced static map with 250m grid...")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot 250m grid (light background)
    for x in x_coords:
        ax.axvline(x=x, color="lightgray", alpha=0.3, linewidth=0.5)
    for y in y_coords:
        ax.axhline(y=y, color="lightgray", alpha=0.3, linewidth=0.5)

    # Color scheme for test types
    test_colors = {"CPT Derived": "red", "HandVane": "orange", "Triaxial": "blue"}

    # Plot test locations by type
    for test_type in ["CPT Derived", "HandVane", "Triaxial"]:
        mask = locations["TestTypes"].str.contains(test_type)
        if mask.any():
            subset = locations[mask]
            ax.scatter(
                subset["Easting"],
                subset["Northing"],
                c=test_colors.get(test_type, "gray"),
                label=f"{test_type} (n={len(subset)})",
                s=50,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

    # Add grid cell labels for cells with tests
    cells_with_tests = grid_cells[grid_cells["TestCount"] > 0]
    for _, cell in cells_with_tests.iterrows():
        ax.text(
            cell["CenterX"],
            cell["CenterY"],
            str(int(cell["TestCount"])),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "yellow", "alpha": 0.7},
        )

    # Styling
    ax.set_xlabel("Easting (m, BNG)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Northing (m, BNG)", fontsize=12, fontweight="bold")
    ax.set_title(
        "SESRO Site: Test Locations with 250m Analysis Grid\n"
        "Grid cells show number of test locations",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    ax.legend(
        title="Test Types",
        title_fontsize=12,
        fontsize=10,
        loc="upper left",
        framealpha=0.9,
    )

    # Add site information
    info_text = (
        f"Site Coverage Analysis:\n"
        f"• Total area: {coverage_stats['total_area_km2']:.1f} km²\n"
        f"• Test locations: {len(locations)}\n"
        f"• Grid cells with tests: {len(cells_with_tests)}\n"
        f"• Mean grid spacing: 250m\n"
        f"• Test density: {coverage_stats['test_density']:.1f}/km²"
    )

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.9},
    )

    # Format axes
    ax.ticklabel_format(style="plain", axis="both")
    ax.grid(True, alpha=0.1)  # Very light grid

    # Save figure
    output_path = output_dir / "test_locations_with_grid.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Enhanced static map saved: {output_path}")
    return output_path


def create_interactive_map_with_grid(locations, grid_cells, coverage_stats, output_dir):
    """Create interactive map with grid overlay using Plotly"""
    print("\n📱 Creating interactive map with grid overlay...")

    # Create figure
    fig = go.Figure()

    # Add grid lines
    grid_lines_x = []
    grid_lines_y = []

    # Get grid bounds
    x_min, x_max = grid_cells["MinX"].min(), grid_cells["MaxX"].max()
    y_min, y_max = grid_cells["MinY"].min(), grid_cells["MaxY"].max()

    # Vertical grid lines
    for x in np.unique(np.concatenate([grid_cells["MinX"], grid_cells["MaxX"]])):
        grid_lines_x.extend([x, x, None])
        grid_lines_y.extend([y_min, y_max, None])

    # Horizontal grid lines
    for y in np.unique(np.concatenate([grid_cells["MinY"], grid_cells["MaxY"]])):
        grid_lines_x.extend([x_min, x_max, None])
        grid_lines_y.extend([y, y, None])

    # Add grid
    fig.add_trace(
        go.Scatter(
            x=grid_lines_x,
            y=grid_lines_y,
            mode="lines",
            line=dict(color="lightgray", width=0.5),
            name="250m Grid",
            showlegend=True,
        )
    )

    # Color scheme for test types
    test_colors = {"CPT Derived": "red", "HandVane": "orange", "Triaxial": "blue"}

    # Add test locations by type
    for test_type in ["CPT Derived", "HandVane", "Triaxial"]:
        mask = locations["TestTypes"].str.contains(test_type)
        if mask.any():
            subset = locations[mask]

            hover_text = [
                f"Location: ({row['Easting']:.0f}, {row['Northing']:.0f})<br>"
                + f"Test Types: {row['TestTypes']}<br>"
                + f"Depth Range: {row['MinDepth']:.1f} - {row['MaxDepth']:.1f} m<br>"
                + f"Mean Cu: {row['MeanCu']:.0f} kPa<br>"
                + f"Total Records: {row['CuRecords']}"
                for _, row in subset.iterrows()
            ]

            fig.add_trace(
                go.Scatter(
                    x=subset["Easting"],
                    y=subset["Northing"],
                    mode="markers",
                    marker=dict(
                        color=test_colors.get(test_type, "gray"),
                        size=8,
                        line=dict(width=1, color="white"),
                    ),
                    name=f"{test_type} (n={len(subset)})",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    # Add grid cell annotations for cells with tests
    cells_with_tests = grid_cells[grid_cells["TestCount"] > 0]
    if len(cells_with_tests) > 0:
        fig.add_trace(
            go.Scatter(
                x=cells_with_tests["CenterX"],
                y=cells_with_tests["CenterY"],
                mode="text",
                text=cells_with_tests["TestCount"].astype(str),
                textfont=dict(size=10, color="black"),
                name="Test Count",
                showlegend=False,
            )
        )

    # Layout
    fig.update_layout(
        title={
            "text": "SESRO Site: Interactive Test Location Map with 250m Grid<br>"
            + f'<span style="font-size:12px">Total: {len(locations)} locations across {coverage_stats["total_area_km2"]:.1f} km²</span>',
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis_title="Easting (m, BNG)",
        yaxis_title="Northing (m, BNG)",
        hovermode="closest",
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )

    # Equal aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    # Save interactive HTML
    output_path = output_dir / "test_locations_interactive_grid.html"
    fig.write_html(output_path)

    print(f"✅ Interactive map saved: {output_path}")
    return output_path


def main():
    """Main execution function"""
    print("🏗️ SESRO 3D Surface Modeling - Enhanced Test Location Analysis")
    print("=" * 70)

    # Setup paths
    base_dir = Path(
        "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100"
    )
    csv_path = base_dir / "CompiledCu.csv"
    output_dir = base_dir / "3d_surface_modeling" / "phase1_data_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    result = load_and_process_data(csv_path)
    if result[0] is None:
        print("❌ Failed to load data. Exiting.")
        return

    _, locations = result

    # Analyze spatial coverage
    coverage_stats = analyze_spatial_coverage(locations)

    # Analyze test types
    _ = analyze_test_types(locations)

    # Generate 250m analysis grid
    grid_cells, x_coords, y_coords = generate_analysis_grid(locations, grid_spacing=250)

    # Assign tests to grid cells
    grid_cells = assign_tests_to_grid(locations, grid_cells)

    # Create enhanced visualizations
    static_path = create_static_map_with_grid(
        locations, grid_cells, x_coords, y_coords, coverage_stats, output_dir
    )
    interactive_path = create_interactive_map_with_grid(
        locations, grid_cells, coverage_stats, output_dir
    )

    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🗺️ Static map: {static_path.name}")
    print(f"📱 Interactive map: {interactive_path.name}")
    print(f"\n🎯 Ready for Phase 1.2: Cu Profile Analysis")


if __name__ == "__main__":
    main()
