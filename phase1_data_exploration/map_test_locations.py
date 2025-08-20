#!/usr/bin/env python3
"""
SESRO 3D Surface Modeling - Phase 1.1: Test Location Mapping
Maps all geotechnical test locations across the site with test type differentiation

Created: August 20, 2025
Purpose: Foundational spatial analysis and coverage assessment
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
import os
import numpy as np
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings("ignore")


def load_and_clean_data(csv_path):
    """Load and perform basic cleaning of geotechnical data"""
    print("=" * 60)
    print("SESRO 3D Surface Modeling: Test Location Mapping")
    print("=" * 60)

    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} total records from CSV")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

    # Basic cleaning - remove invalid coordinates
    df_clean = df[(df["Easting"] > 0) & (df["Northing"] > 0)].copy()
    print(f"✓ {len(df_clean)} records with valid coordinates")

    # Get unique locations
    locations = (
        df_clean.groupby(["Easting", "Northing"])
        .agg(
            {
                "LocationID": "first",
                "TestType": lambda x: "/".join(sorted(set(x))),  # Combine test types
                "Depth": ["min", "max", "count"],
                "AverageCu": ["min", "max", "mean"],
                "Investigation": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    locations.columns = [
        "Easting",
        "Northing",
        "LocationID",
        "TestTypes",
        "MinDepth",
        "MaxDepth",
        "TestCount",
        "MinCu",
        "MaxCu",
        "MeanCu",
        "Investigation",
    ]

    print(f"✓ {len(locations)} unique test locations identified")
    return df_clean, locations


def analyze_spatial_coverage(locations):
    """Analyze spatial distribution and coverage characteristics"""
    print("\n📊 SPATIAL COVERAGE ANALYSIS:")

    # Site boundaries
    easting_range = locations["Easting"].max() - locations["Easting"].min()
    northing_range = locations["Northing"].max() - locations["Northing"].min()
    site_area_km2 = (easting_range * northing_range) / 1000000

    print(f"Site dimensions: {easting_range:.0f}m × {northing_range:.0f}m")
    print(f"Site area: {site_area_km2:.1f} km²")
    print(
        f"Easting range: {locations['Easting'].min():.0f} to {locations['Easting'].max():.0f}"
    )
    print(
        f"Northing range: {locations['Northing'].min():.0f} to {locations['Northing'].max():.0f}"
    )

    # Test density analysis
    avg_spacing = np.sqrt(site_area_km2 * 1000000 / len(locations))
    print(f"Average test spacing: {avg_spacing:.0f}m (assuming uniform distribution)")

    # Calculate actual nearest neighbor distances
    coords = locations[["Easting", "Northing"]].values
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)  # Exclude self-distances
    nearest_distances = np.min(distances, axis=1)

    print(f"Nearest neighbor distances:")
    print(f"  Mean: {nearest_distances.mean():.0f}m")
    print(f"  Median: {np.median(nearest_distances):.0f}m")
    print(f"  Min: {nearest_distances.min():.0f}m")
    print(f"  Max: {nearest_distances.max():.0f}m")

    return {
        "site_area_km2": site_area_km2,
        "avg_spacing": avg_spacing,
        "nearest_distances": nearest_distances,
        "bounds": {
            "easting_min": locations["Easting"].min(),
            "easting_max": locations["Easting"].max(),
            "northing_min": locations["Northing"].min(),
            "northing_max": locations["Northing"].max(),
        },
    }


def generate_grid(bounds, grid_size=250):
    """Generate 250m grid cells across the site"""
    print(f"\n🔲 GENERATING {grid_size}m GRID:")

    # Expand bounds slightly to ensure full coverage
    margin = grid_size * 0.1
    x_min = bounds["easting_min"] - margin
    x_max = bounds["easting_max"] + margin
    y_min = bounds["northing_min"] - margin
    y_max = bounds["northing_max"] + margin

    # Generate grid points
    x_coords = np.arange(x_min, x_max + grid_size, grid_size)
    y_coords = np.arange(y_min, y_max + grid_size, grid_size)

    # Create grid cell centers
    grid_centers = []
    grid_cells = []

    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            # Cell corners
            x1, x2 = x_coords[i], x_coords[i + 1]
            y1, y2 = y_coords[j], y_coords[j + 1]

            # Cell center
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            grid_centers.append([center_x, center_y])

            # Cell boundary
            grid_cells.append(
                {
                    "x_min": x1,
                    "x_max": x2,
                    "y_min": y1,
                    "y_max": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "grid_id": f"G_{i:02d}_{j:02d}",
                }
            )

    print(f"✓ Generated {len(grid_cells)} grid cells ({grid_size}m resolution)")
    print(f"✓ Grid dimensions: {len(x_coords)-1} × {len(y_coords)-1} cells")
    print(f"✓ Total grid area: {((x_max-x_min)*(y_max-y_min))/1000000:.1f} km²")

    return grid_cells, x_coords, y_coords


def assign_tests_to_grid(locations, grid_cells):
    """Assign test locations to grid cells"""
    print("\n📍 ASSIGNING TESTS TO GRID CELLS:")

    # Add grid assignment to locations
    locations["GridID"] = None
    locations["GridCenterX"] = None
    locations["GridCenterY"] = None

    for idx, location in locations.iterrows():
        easting = location["Easting"]
        northing = location["Northing"]

        # Find containing grid cell
        for cell in grid_cells:
            if (
                cell["x_min"] <= easting < cell["x_max"]
                and cell["y_min"] <= northing < cell["y_max"]
            ):
                locations.at[idx, "GridID"] = cell["grid_id"]
                locations.at[idx, "GridCenterX"] = cell["center_x"]
                locations.at[idx, "GridCenterY"] = cell["center_y"]
                break

    # Analyze grid occupancy
    occupied_cells = locations["GridID"].dropna().nunique()
    empty_cells = len(grid_cells) - occupied_cells

    print(f"✓ Grid cells with tests: {occupied_cells}")
    print(f"✓ Empty grid cells: {empty_cells}")
    print(f"✓ Grid occupancy rate: {occupied_cells/len(grid_cells)*100:.1f}%")

    # Multiple tests per cell analysis
    multi_test_cells = locations.groupby("GridID").size()
    multi_test_cells = multi_test_cells[multi_test_cells > 1]

    if len(multi_test_cells) > 0:
        print(f"✓ Cells with multiple tests: {len(multi_test_cells)}")
        print(f"  Max tests per cell: {multi_test_cells.max()}")
        print(
            f"  Mean tests per occupied cell: {locations.groupby('GridID').size().mean():.1f}"
        )

    return locations


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
    """Create high-quality static map of test locations"""
    print("\n🗺️  Creating static location map...")

    fig, ax = plt.subplots(figsize=(16, 12))

    # Color mapping for test types
    test_colors = {
        "CPTDerived": "green",
        "HandVane": "orange",
        "TriaxialTotal": "blue",
        "Mixed": "red",
    }

    # Determine primary test type for coloring
    locations["PrimaryTestType"] = locations["TestTypes"].str.split("/").str[0]
    locations["PrimaryTestType"] = locations["PrimaryTestType"].where(
        ~locations["TestTypes"].str.contains("/"), "Mixed"
    )

    # Plot locations by test type
    for test_type in locations["PrimaryTestType"].unique():
        subset = locations[locations["PrimaryTestType"] == test_type]
        ax.scatter(
            subset["Easting"],
            subset["Northing"],
            c=test_colors.get(test_type, "gray"),
            label=f"{test_type} (n={len(subset)})",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Add site boundary box
    bounds = coverage_stats["bounds"]
    ax.plot(
        [
            bounds["easting_min"],
            bounds["easting_max"],
            bounds["easting_max"],
            bounds["easting_min"],
            bounds["easting_min"],
        ],
        [
            bounds["northing_min"],
            bounds["northing_min"],
            bounds["northing_max"],
            bounds["northing_max"],
            bounds["northing_min"],
        ],
        "k--",
        linewidth=2,
        alpha=0.8,
        label="Site Boundary",
    )

    # Formatting
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    ax.set_title(
        "SESRO: Geotechnical Test Location Map\n3D Surface Modeling Phase 1",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_aspect("equal")

    # Add statistics text
    stats_text = (
        f"Site Coverage Analysis:\n"
        f"Total Locations: {len(locations)}\n"
        f"Site Area: {coverage_stats['site_area_km2']:.1f} km²\n"
        f"Avg Spacing: {coverage_stats['avg_spacing']:.0f}m\n"
        f"Mean NN Distance: {coverage_stats['nearest_distances'].mean():.0f}m"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Save static map
    static_path = os.path.join(output_dir, "test_locations_map.png")
    plt.savefig(static_path, dpi=300, bbox_inches="tight")
    print(f"✓ Static map saved: {static_path}")

    plt.show()
    return static_path


def create_interactive_map(locations, coverage_stats, output_dir):
    """Create interactive HTML map of test locations"""
    print("🌐 Creating interactive location map...")

    # Determine primary test type for coloring
    locations["PrimaryTestType"] = locations["TestTypes"].str.split("/").str[0]
    locations["PrimaryTestType"] = locations["PrimaryTestType"].where(
        ~locations["TestTypes"].str.contains("/"), "Mixed"
    )

    # Color mapping
    color_map = {
        "CPTDerived": "green",
        "HandVane": "orange",
        "TriaxialTotal": "blue",
        "Mixed": "red",
    }

    # Create figure
    fig = go.Figure()

    # Add traces for each test type
    for test_type in locations["PrimaryTestType"].unique():
        subset = locations[locations["PrimaryTestType"] == test_type]

        fig.add_trace(
            go.Scatter(
                x=subset["Easting"],
                y=subset["Northing"],
                mode="markers",
                name=f"{test_type} (n={len(subset)})",
                marker={
                    "color": color_map.get(test_type, "gray"),
                    "size": 8,
                    "opacity": 0.8,
                    "line": {"width": 1, "color": "black"},
                },
                hovertemplate="<b>%{text}</b><br>"
                + "Easting: %{x:.0f}m<br>"
                + "Northing: %{y:.0f}m<br>"
                + "Test Types: %{customdata[0]}<br>"
                + "Test Count: %{customdata[1]}<br>"
                + "Depth Range: %{customdata[2]:.1f} - %{customdata[3]:.1f}m<br>"
                + "Cu Range: %{customdata[4]:.0f} - %{customdata[5]:.0f} kPa<br>"
                + "<extra></extra>",
                text=subset["LocationID"],
                customdata=np.column_stack(
                    (
                        subset["TestTypes"],
                        subset["TestCount"],
                        subset["MinDepth"],
                        subset["MaxDepth"],
                        subset["MinCu"],
                        subset["MaxCu"],
                    )
                ),
            )
        )

    # Add site boundary
    bounds = coverage_stats["bounds"]
    fig.add_trace(
        go.Scatter(
            x=[
                bounds["easting_min"],
                bounds["easting_max"],
                bounds["easting_max"],
                bounds["easting_min"],
                bounds["easting_min"],
            ],
            y=[
                bounds["northing_min"],
                bounds["northing_min"],
                bounds["northing_max"],
                bounds["northing_max"],
                bounds["northing_min"],
            ],
            mode="lines",
            name="Site Boundary",
            line={"color": "black", "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "SESRO: Interactive Geotechnical Test Location Map<br>"
            + "<sub>3D Surface Modeling Phase 1 - Hover for Details</sub>",
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis_title="Easting (m)",
        yaxis_title="Northing (m)",
        width=1200,
        height=900,
        template="plotly_white",
        showlegend=True,
        legend={"x": 0.02, "y": 0.98},
        hovermode="closest",
    )

    # Set equal aspect ratio
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Save interactive map
    interactive_path = os.path.join(output_dir, "test_locations_interactive.html")
    fig.write_html(interactive_path)
    print(f"✓ Interactive map saved: {interactive_path}")

    return interactive_path


def main():
    """Main execution function"""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_dir, "CompiledCu.csv")
    output_dir = os.path.join(os.path.dirname(script_dir), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load and analyze data
    result = load_and_clean_data(csv_path)
    if result is None:
        return

    df_clean, locations = result

    # Perform analyses
    coverage_stats = analyze_spatial_coverage(locations)
    test_type_stats = analyze_test_types(locations)

    # Create visualizations
    static_map_path = create_static_map(locations, coverage_stats, output_dir)
    interactive_map_path = create_interactive_map(locations, coverage_stats, output_dir)

    # Save location data for subsequent phases
    locations_csv_path = os.path.join(output_dir, "unique_test_locations.csv")
    locations.to_csv(locations_csv_path, index=False)
    print(f"✓ Location data saved: {locations_csv_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 1.1 COMPLETE: Test Location Mapping")
    print("=" * 60)
    print(f"✓ Analyzed {len(locations)} unique test locations")
    print(f"✓ Site coverage: {coverage_stats['site_area_km2']:.1f} km²")
    print(f"✓ Static map: {os.path.basename(static_map_path)}")
    print(f"✓ Interactive map: {os.path.basename(interactive_map_path)}")
    print(f"✓ Location dataset: {os.path.basename(locations_csv_path)}")
    print("\nReady for Phase 1.2: Cu Profile Analysis")


if __name__ == "__main__":
    main()
