#!/usr/bin/env python3
"""
Depth Interpolation for Cu = 100 kPa Analysis
Creates grid visualization showing depth values where undrained shear strength reaches 100 kPa

Uses scipy.interpolate.griddata for spatial interpolation of depth values across quadrants
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthInterpolationGrid:
    """
    Creates depth interpolation visualization for Cu = 100 kPa analysis
    """

    def __init__(self, csv_path):
        """Initialize with CSV data path"""
        self.csv_path = csv_path
        self.df = None
        self.data_quadrants = None
        self.grid_bounds = None

    def load_data(self):
        """Load and prepare grid quadrant data with depths"""
        logger.info(f"Loading grid data from: {self.csv_path}")

        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} total quadrants")

        # Filter to quadrants with data and depths
        self.data_quadrants = self.df[
            (self.df["has_data"] == 1) & (self.df["sample_depth"].notna())
        ].copy()

        logger.info(f"Found {len(self.data_quadrants)} quadrants with depth data")

        # Calculate grid bounds
        self.grid_bounds = {
            "easting_min": self.df["easting_min"].min(),
            "easting_max": self.df["easting_max"].max(),
            "northing_min": self.df["northing_min"].min(),
            "northing_max": self.df["northing_max"].max(),
            "rows": self.df["row"].max() + 1,
            "cols": self.df["col"].max() + 1,
        }

        logger.info(
            f"Grid dimensions: {self.grid_bounds['cols']} x {self.grid_bounds['rows']}"
        )

        return self.data_quadrants

    def perform_interpolation(self, method="cubic"):
        """
        Perform spatial interpolation of depth values

        Args:
            method (str): Interpolation method ('linear', 'cubic', 'nearest')
        """
        logger.info(f"Performing depth interpolation using {method} method...")

        if self.data_quadrants is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Prepare known points (x, y, depth)
        known_points = self.data_quadrants[["center_easting", "center_northing"]].values
        known_depths = self.data_quadrants["sample_depth"].values

        logger.info(f"Interpolating from {len(known_points)} known depth points")
        logger.info(
            f"Known depth range: {known_depths.min():.1f} - {known_depths.max():.1f}m"
        )

        # Create target grid for all quadrants
        target_points = self.df[["center_easting", "center_northing"]].values

        # Perform interpolation
        interpolated_depths = griddata(
            known_points,
            known_depths,
            target_points,
            method=method,
            fill_value=np.nan,  # Use NaN for points outside convex hull
        )

        # Add interpolated depths to dataframe
        self.df = self.df.copy()
        self.df["interpolated_depth"] = interpolated_depths

        # Count successful interpolations
        valid_interpolations = np.isfinite(interpolated_depths).sum()
        logger.info(
            f"Successfully interpolated {valid_interpolations}/{len(self.df)} quadrants"
        )

        # Create final depth grid (use known depths where available, interpolated elsewhere)
        self.df["final_depth"] = self.df["sample_depth"].fillna(
            self.df["interpolated_depth"]
        )

        return self.df

    def create_depth_grid_visualization(
        self, output_path, show_values=True, colormap="viridis"
    ):
        """
        Create grid visualization showing depth values

        Args:
            output_path (str): Path to save the visualization
            show_values (bool): Whether to show depth values in each quadrant
            colormap (str): Matplotlib colormap to use
        """
        logger.info(f"Creating depth grid visualization: {output_path}")

        if "final_depth" not in self.df.columns:
            raise ValueError(
                "Interpolation not performed. Call perform_interpolation() first."
            )

        # Prepare data for visualization
        rows, cols = self.grid_bounds["rows"], self.grid_bounds["cols"]

        # Create depth matrix
        depth_matrix = np.full((rows, cols), np.nan)
        has_data_matrix = np.zeros((rows, cols), dtype=bool)

        for _, row_data in self.df.iterrows():
            r, c = int(row_data["row"]), int(row_data["col"])
            depth_matrix[r, c] = row_data["final_depth"]
            has_data_matrix[r, c] = row_data["has_data"] == 1

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Define depth range for consistent coloring
        valid_depths = depth_matrix[np.isfinite(depth_matrix)]
        if len(valid_depths) > 0:
            vmin, vmax = valid_depths.min(), valid_depths.max()
            logger.info(f"Depth range for visualization: {vmin:.1f} - {vmax:.1f}m")
        else:
            vmin, vmax = 0, 20  # Default range

        # Draw grid squares
        for row in range(rows):
            for col in range(cols):
                depth = depth_matrix[row, col]
                has_data = has_data_matrix[row, col]

                if np.isfinite(depth):
                    # Use color based on depth
                    cmap = plt.cm.get_cmap(colormap)
                    norm_depth = (depth - vmin) / (vmax - vmin) if vmax > vmin else 0
                    color = cmap(norm_depth)
                    alpha = 1.0
                    edge_color = "red" if has_data else "black"
                    edge_width = 2.0 if has_data else 0.5
                else:
                    # No data - white
                    color = "white"
                    alpha = 1.0
                    edge_color = "gray"
                    edge_width = 0.3

                # Create square
                rect = patches.Rectangle(
                    (col, row),
                    1,
                    1,
                    linewidth=edge_width,
                    edgecolor=edge_color,
                    facecolor=color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

                # Add depth text if requested and valid
                if show_values and np.isfinite(depth):
                    text_color = "white" if norm_depth > 0.5 else "black"
                    fontweight = "bold" if has_data else "normal"
                    fontsize = 8 if has_data else 6

                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{depth:.1f}",
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        fontweight=fontweight,
                        color=text_color,
                    )

        # Set axis properties
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Match standard grid orientation

        # Add colorbar
        if len(valid_depths) > 0:
            sm = plt.cm.ScalarMappable(
                cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label(
                "Depth where Cu = 100 kPa (m)", fontsize=12, fontweight="bold"
            )

        # Add title and stats
        known_count = len(self.data_quadrants)
        interpolated_count = np.isfinite(self.df["interpolated_depth"]).sum()
        total_with_depth = np.isfinite(self.df["final_depth"]).sum()

        title = "Depth Interpolation for Cu = 100 kPa Analysis\\n"
        title += f"Known depths: {known_count} quadrants (red borders)\\n"
        title += f"Interpolated depths: {interpolated_count} quadrants\\n"
        title += f"Total coverage: {total_with_depth}/{len(self.df)} quadrants"

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

        # Remove axis ticks for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])

        # Save the plot
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        logger.info(f"Depth grid visualization saved: {output_path}")

    def export_interpolated_data(self, output_dir="."):
        """Export interpolated results to CSV"""
        if "final_depth" not in self.df.columns:
            raise ValueError(
                "Interpolation not performed. Call perform_interpolation() first."
            )

        # Export complete grid with interpolated depths
        output_path = Path(output_dir) / "interpolated_depth_grid.csv"
        self.df.to_csv(output_path, index=False)
        logger.info(f"Interpolated data exported: {output_path}")

        # Export summary statistics
        summary_data = {
            "total_quadrants": len(self.df),
            "known_depths": len(self.data_quadrants),
            "interpolated_depths": np.isfinite(self.df["interpolated_depth"]).sum(),
            "total_with_depth": np.isfinite(self.df["final_depth"]).sum(),
            "min_known_depth": self.data_quadrants["sample_depth"].min(),
            "max_known_depth": self.data_quadrants["sample_depth"].max(),
            "mean_known_depth": self.data_quadrants["sample_depth"].mean(),
            "min_interpolated_depth": np.nanmin(self.df["final_depth"]),
            "max_interpolated_depth": np.nanmax(self.df["final_depth"]),
            "mean_interpolated_depth": np.nanmean(self.df["final_depth"]),
        }

        summary_df = pd.DataFrame([summary_data])
        summary_path = Path(output_dir) / "depth_interpolation_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary statistics exported: {summary_path}")

        return output_path, summary_path


def main():
    """Main execution function"""
    print("DEPTH INTERPOLATION FOR Cu = 100 kPa ANALYSIS")
    print("=" * 50)

    # Paths
    csv_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_quadrants.csv"
    output_dir = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs"

    try:
        # Initialize depth interpolation
        depth_interpolator = DepthInterpolationGrid(csv_path)

        # Load data
        data_quadrants = depth_interpolator.load_data()

        # Perform interpolation
        interpolated_df = depth_interpolator.perform_interpolation(method="cubic")

        # Create visualization
        viz_path = Path(output_dir) / "depth_interpolation_grid.png"
        depth_interpolator.create_depth_grid_visualization(
            str(viz_path), show_values=True, colormap="viridis"
        )

        # Export results
        data_path, stats_path = depth_interpolator.export_interpolated_data(output_dir)

        print("\\n" + "=" * 50)
        print("DEPTH INTERPOLATION COMPLETE")
        print("=" * 50)
        print("✅ Spatial interpolation performed using cubic method")
        print("✅ Grid visualization created with depth values")
        print("✅ Known depths marked with red borders")
        print("✅ Interpolated data exported to CSV")
        print("\\n📊 Results:")
        print(f"   • Known depth points: {len(data_quadrants)}")
        print(
            f"   • Interpolated quadrants: {np.isfinite(interpolated_df['interpolated_depth']).sum()}"
        )
        print(
            f"   • Total coverage: {np.isfinite(interpolated_df['final_depth']).sum()}/{len(interpolated_df)}"
        )
        print(
            f"   • Depth range: {data_quadrants['sample_depth'].min():.1f} - {data_quadrants['sample_depth'].max():.1f}m"
        )
        print(f"\\n📁 Files created:")
        print(f"   • {viz_path}")
        print(f"   • {data_path}")
        print(f"   • {stats_path}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
