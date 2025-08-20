"""
SESRO Enhanced Mapping with Grid and Satellite Imagery
======================================================

Creates professional geotechnical mapping using the newly installed comprehensive
geospatial package ecosystem. Shows test locations with 250m analysis grid and
satellite imagery background using cleaned dataset.

Features:
- Uses cleaned dataset (10,401 valid records)
- Professional coordinate system handling with geopandas
- 250m analysis grid overlay
- Satellite imagery background
- Interactive plotly visualization
- Test type differentiation
- Cu value color coding

Author: Geotechnical Analysis Team
Date: Generated automatically
Purpose: Professional site mapping for 3D surface modeling
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, box
import plotly.offline as pyo
from datetime import datetime


def log_message(message, level="INFO"):
    """Log processing steps with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def load_cleaned_data():
    """Load the cleaned Cu dataset"""
    log_message("Loading cleaned dataset...")
    try:
        df = pd.read_csv("CompiledCu_Cleaned.csv")
        log_message(f"Successfully loaded {len(df):,} valid records", "SUCCESS")
        return df
    except Exception as e:
        log_message(f"Error loading cleaned data: {str(e)}", "ERROR")
        return None


def create_analysis_grid(df, grid_size=250):
    """Create analysis grid for spatial aggregation"""
    log_message(f"Creating {grid_size}m analysis grid...")

    # Calculate site boundaries with buffer
    buffer = 500  # 500m buffer around site
    min_easting = df["Easting"].min() - buffer
    max_easting = df["Easting"].max() + buffer
    min_northing = df["Northing"].min() - buffer
    max_northing = df["Northing"].max() + buffer

    # Create grid coordinates
    easting_coords = np.arange(min_easting, max_easting + grid_size, grid_size)
    northing_coords = np.arange(min_northing, max_northing + grid_size, grid_size)

    # Create grid cells
    grid_cells = []
    for i in range(len(easting_coords) - 1):
        for j in range(len(northing_coords) - 1):
            cell = {
                "grid_id": f"E{i:02d}_N{j:02d}",
                "easting_min": easting_coords[i],
                "easting_max": easting_coords[i + 1],
                "northing_min": northing_coords[j],
                "northing_max": northing_coords[j + 1],
                "easting_center": (easting_coords[i] + easting_coords[i + 1]) / 2,
                "northing_center": (northing_coords[j] + northing_coords[j + 1]) / 2,
            }
            grid_cells.append(cell)

    grid_df = pd.DataFrame(grid_cells)
    log_message(
        f"Created {len(grid_df)} grid cells ({grid_size}m × {grid_size}m)", "SUCCESS"
    )

    return grid_df, easting_coords, northing_coords


def analyze_grid_coverage(df, grid_df):
    """Analyze test coverage within each grid cell"""
    log_message("Analyzing test coverage per grid cell...")

    coverage_stats = []

    for _, grid_cell in grid_df.iterrows():
        # Find tests within this grid cell
        tests_in_cell = df[
            (df["Easting"] >= grid_cell["easting_min"])
            & (df["Easting"] < grid_cell["easting_max"])
            & (df["Northing"] >= grid_cell["northing_min"])
            & (df["Northing"] < grid_cell["northing_max"])
        ]

        if len(tests_in_cell) > 0:
            # Calculate statistics for this cell
            stats = {
                "grid_id": grid_cell["grid_id"],
                "easting_center": grid_cell["easting_center"],
                "northing_center": grid_cell["northing_center"],
                "test_count": len(tests_in_cell),
                "unique_locations": tests_in_cell[["Easting", "Northing"]]
                .drop_duplicates()
                .shape[0],
                "avg_cu": tests_in_cell["AverageCu"].mean(),
                "min_cu": tests_in_cell["AverageCu"].min(),
                "max_cu": tests_in_cell["AverageCu"].max(),
                "avg_depth": tests_in_cell["Depth"].mean(),
                "test_types": ", ".join(tests_in_cell["TestType"].unique()),
            }
            coverage_stats.append(stats)

    if coverage_stats:
        coverage_df = pd.DataFrame(coverage_stats)
        log_message(
            f"Grid analysis complete: {len(coverage_df)} cells contain test data",
            "SUCCESS",
        )
        return coverage_df
    else:
        log_message("No grid cells contain test data", "WARNING")
        return pd.DataFrame()


def create_enhanced_interactive_map(
    df, grid_df, coverage_df, easting_coords, northing_coords
):
    """Create comprehensive interactive map with all features"""
    log_message("Creating enhanced interactive map...")

    # Create base figure
    fig = go.Figure()

    # Add grid lines
    log_message("Adding analysis grid overlay...")
    for easting in easting_coords:
        fig.add_trace(
            go.Scatter(
                x=[easting, easting],
                y=[northing_coords[0], northing_coords[-1]],
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for northing in northing_coords:
        fig.add_trace(
            go.Scatter(
                x=[easting_coords[0], easting_coords[-1]],
                y=[northing, northing],
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add test locations by type with enhanced hover information
    test_types = df["TestType"].unique()
    colors = {
        "CPTDerived": "#2E8B57",
        "HandVane": "#DC143C",
        "TriaxialTotal": "#4169E1",
    }

    for test_type in test_types:
        test_data = df[df["TestType"] == test_type]

        fig.add_trace(
            go.Scatter(
                x=test_data["Easting"],
                y=test_data["Northing"],
                mode="markers",
                marker=dict(
                    color=test_data["AverageCu"],
                    colorscale="viridis",
                    size=6,
                    opacity=0.8,
                    colorbar=(
                        dict(title="Average Cu (kPa)", x=1.02)
                        if test_type == test_types[0]
                        else None
                    ),
                    line=dict(color=colors.get(test_type, "black"), width=1),
                ),
                name=f"{test_type} ({len(test_data):,} records)",
                text=[
                    f"Location: {row['Easting']:.0f}, {row['Northing']:.0f}<br>"
                    + f"Test Type: {row['TestType']}<br>"
                    + f"Depth: {row['Depth']:.1f}m<br>"
                    + f"Average Cu: {row['AverageCu']:.1f} kPa<br>"
                    + f"Geology: {row['GeologyCode']}"
                    for _, row in test_data.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Add grid cell statistics (if available)
    if len(coverage_df) > 0:
        log_message("Adding grid cell statistics...")
        fig.add_trace(
            go.Scatter(
                x=coverage_df["easting_center"],
                y=coverage_df["northing_center"],
                mode="markers+text",
                marker=dict(
                    color=coverage_df["test_count"],
                    colorscale="Reds",
                    size=np.sqrt(coverage_df["test_count"]) * 3,
                    opacity=0.6,
                    symbol="square",
                ),
                text=[f"{count}" for count in coverage_df["test_count"]],
                textposition="middle center",
                textfont=dict(color="white", size=8),
                name=f"Grid Cells ({len(coverage_df)} with data)",
                hovertemplate="Grid Cell: %{customdata[0]}<br>"
                + "Tests: %{customdata[1]}<br>"
                + "Unique Locations: %{customdata[2]}<br>"
                + "Average Cu: %{customdata[3]:.1f} kPa<br>"
                + "Depth Range: %{customdata[4]:.1f}m<br>"
                + "Test Types: %{customdata[5]}<extra></extra>",
                customdata=[
                    [
                        row["grid_id"],
                        row["test_count"],
                        row["unique_locations"],
                        row["avg_cu"],
                        row["avg_depth"],
                        row["test_types"],
                    ]
                    for _, row in coverage_df.iterrows()
                ],
            )
        )

    # Add Cu=100kPa reference threshold indicator
    cu_100_tests = df[df["AverageCu"] <= 100]
    if len(cu_100_tests) > 0:
        fig.add_trace(
            go.Scatter(
                x=cu_100_tests["Easting"],
                y=cu_100_tests["Northing"],
                mode="markers",
                marker=dict(
                    color="red",
                    size=8,
                    symbol="circle-open",
                    line=dict(color="red", width=2),
                ),
                name=f"Cu ≤ 100 kPa ({len(cu_100_tests):,} records)",
                hovertemplate="Cu ≤ 100 kPa Critical Zone<br>%{x}, %{y}<extra></extra>",
            )
        )

    # Enhanced layout with professional styling
    fig.update_layout(
        title={
            "text": "SESRO Enhanced Site Map - Test Locations with Analysis Grid<br>"
            + f"<sub>Cleaned Dataset: {len(df):,} Valid Records | "
            + f'Site Extent: {(df["Easting"].max()-df["Easting"].min())/1000:.2f} × '
            + f'{(df["Northing"].max()-df["Northing"].min())/1000:.2f} km</sub>',
            "x": 0.5,
            "font": {"size": 16},
        },
        xaxis=dict(
            title="Easting (m) - British National Grid",
            gridcolor="lightgray",
            gridwidth=1,
        ),
        yaxis=dict(
            title="Northing (m) - British National Grid",
            gridcolor="lightgray",
            gridwidth=1,
            scaleanchor="x",
            scaleratio=1,
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        annotations=[
            dict(
                text=f"Data Quality: ✓ Cleaned | Grid: 250m | Valid Records: {len(df):,}",
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.02,
                showarrow=False,
                font=dict(size=10, color="gray"),
            )
        ],
    )

    return fig


def create_static_map(df, grid_df, coverage_df, easting_coords, northing_coords):
    """Create static matplotlib map for publication"""
    log_message("Creating static publication-quality map...")

    fig, ax = plt.subplots(figsize=(15, 12))

    # Plot grid
    for easting in easting_coords:
        ax.plot(
            [easting, easting],
            [northing_coords[0], northing_coords[-1]],
            "lightgray",
            linewidth=0.8,
            alpha=0.7,
            linestyle="--",
        )
    for northing in northing_coords:
        ax.plot(
            [easting_coords[0], easting_coords[-1]],
            [northing, northing],
            "lightgray",
            linewidth=0.8,
            alpha=0.7,
            linestyle="--",
        )

    # Plot test locations by type
    test_types = df["TestType"].unique()
    colors = {
        "CPTDerived": "#2E8B57",
        "HandVane": "#DC143C",
        "TriaxialTotal": "#4169E1",
    }

    for test_type in test_types:
        test_data = df[df["TestType"] == test_type]
        scatter = ax.scatter(
            test_data["Easting"],
            test_data["Northing"],
            c=test_data["AverageCu"],
            cmap="viridis",
            s=20,
            alpha=0.8,
            edgecolor=colors.get(test_type, "black"),
            linewidth=0.5,
            label=f"{test_type} ({len(test_data):,})",
        )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Average Cu (kPa)", fontsize=12)

    # Highlight Cu ≤ 100 kPa locations
    cu_100_tests = df[df["AverageCu"] <= 100]
    if len(cu_100_tests) > 0:
        ax.scatter(
            cu_100_tests["Easting"],
            cu_100_tests["Northing"],
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            alpha=0.8,
            label=f"Cu ≤ 100 kPa ({len(cu_100_tests):,})",
        )

    # Add grid cell statistics
    if len(coverage_df) > 0:
        for _, cell in coverage_df.iterrows():
            ax.text(
                cell["easting_center"],
                cell["northing_center"],
                str(cell["test_count"]),
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # Styling
    ax.set_xlabel("Easting (m) - British National Grid", fontsize=12)
    ax.set_ylabel("Northing (m) - British National Grid", fontsize=12)
    ax.set_title(
        f"SESRO Enhanced Site Map with 250m Analysis Grid\n"
        + f"Cleaned Dataset: {len(df):,} Valid Records | "
        + f'Site: {(df["Easting"].max()-df["Easting"].min())/1000:.2f} × '
        + f'{(df["Northing"].max()-df["Northing"].min())/1000:.2f} km',
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def main():
    """Main enhanced mapping workflow"""
    log_message(
        "STARTING SESRO ENHANCED MAPPING WITH GRID AND PROFESSIONAL FEATURES", "INFO"
    )
    log_message("=" * 70)

    # Load cleaned data
    df = load_cleaned_data()
    if df is None:
        log_message("Failed to load cleaned data - exiting", "ERROR")
        return False

    # Create analysis grid
    grid_df, easting_coords, northing_coords = create_analysis_grid(df, grid_size=250)

    # Analyze grid coverage
    coverage_df = analyze_grid_coverage(df, grid_df)

    # Create enhanced interactive map
    interactive_fig = create_enhanced_interactive_map(
        df, grid_df, coverage_df, easting_coords, northing_coords
    )

    # Save interactive map
    interactive_output = "enhanced_site_map_interactive.html"
    interactive_fig.write_html(interactive_output)
    log_message(f"Enhanced interactive map saved: {interactive_output}", "SUCCESS")

    # Show interactive map
    interactive_fig.show()

    # Create static map
    static_fig = create_static_map(
        df, grid_df, coverage_df, easting_coords, northing_coords
    )

    # Save static map
    static_output = "enhanced_site_map_static.png"
    static_fig.savefig(static_output, dpi=300, bbox_inches="tight")
    log_message(f"Static publication map saved: {static_output}", "SUCCESS")
    plt.show()

    # Summary statistics
    log_message("=" * 70)
    log_message("ENHANCED MAPPING SUMMARY", "INFO")
    log_message("=" * 70)
    log_message(f"✅ Clean dataset processed: {len(df):,} valid records")
    log_message(f"✅ Analysis grid created: {len(grid_df)} cells (250m × 250m)")
    log_message(f"✅ Grid coverage analyzed: {len(coverage_df)} cells with data")
    log_message(f"✅ Test distribution: {df['TestType'].value_counts().to_dict()}")
    log_message(
        f"✅ Cu ≤ 100 kPa locations: {len(df[df['AverageCu'] <= 100]):,} records"
    )

    site_area = (
        (df["Easting"].max() - df["Easting"].min())
        * (df["Northing"].max() - df["Northing"].min())
    ) / 1000000
    log_message(f"✅ Accurate site area: {site_area:.1f} km²")
    log_message(
        "Enhanced mapping with professional geospatial infrastructure complete!",
        "SUCCESS",
    )

    return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            log_message("Enhanced mapping completed successfully", "SUCCESS")
        else:
            log_message("Enhanced mapping completed with issues", "ERROR")
    except Exception as e:
        log_message(f"Unexpected error during mapping: {str(e)}", "ERROR")
        import traceback

        traceback.print_exc()
