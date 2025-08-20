#!/usr/bin/env python3
"""
Depth Grid with Interpolation
Creates a grid showing both known and interpolated depth values
Based on simple_depth_grid_from_csv.py but adds spatial interpolation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthGridWithInterpolation:
    """
    Creates a grid visualization showing both known and interpolated depth values
    """

    def __init__(self, csv_path):
        """Initialize with CSV data path"""
        self.csv_path = csv_path
        self.df = None
        self.interpolated_grid = None

    def load_csv_data(self):
        """Load the grid quadrants CSV with depth data"""
        logger.info(f"Loading grid data from: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} quadrants")

        # Count quadrants with depth data
        with_depth = self.df[self.df["sample_depth"].notna()]
        logger.info(f"Found {len(with_depth)} quadrants with known depth values")

        return self.df

    def perform_interpolation(self, method="linear"):
        """
        Perform spatial interpolation to fill empty quadrants

        Args:
            method (str): Interpolation method ('linear', 'nearest', 'cubic')
        """
        logger.info(f"Performing interpolation using {method} method...")

        # Extract known data points
        known_data = self.df[self.df["sample_depth"].notna()].copy()
        known_points = known_data[["center_easting", "center_northing"]].values
        known_depths = known_data["sample_depth"].values

        logger.info(f"Interpolating from {len(known_points)} known depth points")
        logger.info(
            f"Known depth range: {known_depths.min():.1f} - {known_depths.max():.1f}m"
        )

        # Prepare all grid points for interpolation
        all_points = self.df[["center_easting", "center_northing"]].values

        # Perform interpolation
        interpolated_depths = griddata(
            known_points, known_depths, all_points, method=method, fill_value=np.nan
        )

        # Add interpolated values to dataframe
        self.df = self.df.copy()
        self.df["interpolated_depth"] = interpolated_depths

        # Create final depth column (known values take precedence over interpolated)
        self.df["final_depth"] = self.df["sample_depth"].fillna(
            self.df["interpolated_depth"]
        )

        # Count results
        total_interpolated = np.isfinite(self.df["interpolated_depth"]).sum()
        total_with_final_depth = np.isfinite(self.df["final_depth"]).sum()
        known_count = len(known_data)

        logger.info(f"Interpolation results:")
        logger.info(f"  Known depths: {known_count}")
        logger.info(
            f"  Interpolated: {total_interpolated - known_count}"
        )  # Subtract known from total
        logger.info(f"  Total coverage: {total_with_final_depth}/{len(self.df)}")

        return self.df

    def create_depth_grid_with_interpolation(self, output_path):
        """
        Create grid showing both known and interpolated depth values

        Args:
            output_path (str): Path to save the output PNG
        """
        logger.info(f"Creating depth grid with interpolation: {output_path}")

        if "final_depth" not in self.df.columns:
            raise ValueError(
                "Interpolation not performed. Call perform_interpolation() first."
            )

        # Get grid dimensions
        num_rows = self.df["row"].max() + 1
        num_cols = self.df["col"].max() + 1

        # Create figure with white background
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Count different types of quadrants
        known_count = 0
        interpolated_count = 0
        no_data_count = 0

        # Draw grid squares
        for _, row_data in self.df.iterrows():
            row = int(row_data["row"])
            col = int(row_data["col"])
            known_depth = row_data["sample_depth"]
            final_depth = row_data["final_depth"]

            # Determine color, text, and styling based on data type
            if pd.notna(known_depth):
                # Known depth data - blue background
                color = "lightblue"
                text_color = "black"
                text = f"{known_depth:.1f}"
                border_color = "blue"
                border_width = 2.0
                known_count += 1
            elif pd.notna(final_depth):
                # Interpolated depth - light green background
                color = "lightgreen"
                text_color = "black"
                text = f"{final_depth:.1f}"
                border_color = "green"
                border_width = 1.0
                interpolated_count += 1
            else:
                # No data - white background
                color = "white"
                text_color = "gray"
                text = ""
                border_color = "lightgray"
                border_width = 0.5
                no_data_count += 1

            # Create square
            rect = patches.Rectangle(
                (col, row),
                1,
                1,
                linewidth=border_width,
                edgecolor=border_color,
                facecolor=color,
                alpha=1.0,
            )
            ax.add_patch(rect)

            # Add depth text if available
            if text:
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if pd.notna(known_depth) else "normal",
                    color=text_color,
                )

        # Set axis properties
        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        ax.set_aspect("equal")

        # Remove axis labels and ticks for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Add title with legend
        title = "SESRO Depth Values with Interpolation\\n"
        title += f"Known: {known_count} (Blue), Interpolated: {interpolated_count} (Green), No Data: {no_data_count} (White)\\n"
        title += "Blue = Known Depths, Green = Interpolated Depths, White = No Coverage"

        plt.suptitle(title, fontsize=13, fontweight="bold", y=0.95)

        # Save the grid
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        logger.info(f"Depth grid with interpolation saved: {output_path}")
        logger.info(f"Grid dimensions: {num_cols} x {num_rows}")
        logger.info(f"Blue squares (known): {known_count}")
        logger.info(f"Green squares (interpolated): {interpolated_count}")
        logger.info(f"White squares (no data): {no_data_count}")

    def export_interpolated_csv(self, output_path):
        """Export the complete grid with interpolated values to CSV"""
        if "final_depth" not in self.df.columns:
            raise ValueError(
                "Interpolation not performed. Call perform_interpolation() first."
            )

        # Create clean export DataFrame
        export_df = self.df[
            [
                "quadrant_id",
                "row",
                "col",
                "has_data",
                "center_easting",
                "center_northing",
                "sample_depth",
                "interpolated_depth",
                "final_depth",
            ]
        ].copy()

        # Add helpful flags
        export_df["is_known"] = export_df["sample_depth"].notna()
        export_df["is_interpolated"] = (
            export_df["sample_depth"].isna() & export_df["interpolated_depth"].notna()
        )
        export_df["has_final_depth"] = export_df["final_depth"].notna()

        # Export to CSV
        export_df.to_csv(output_path, index=False)
        logger.info(f"Interpolated data exported to: {output_path}")

        # Print summary
        logger.info(f"Export summary:")
        logger.info(f"  Total quadrants: {len(export_df)}")
        logger.info(f"  Known depths: {export_df['is_known'].sum()}")
        logger.info(f"  Interpolated depths: {export_df['is_interpolated'].sum()}")
        logger.info(f"  Total with depth: {export_df['has_final_depth'].sum()}")

        return output_path


def main():
    """Main execution function"""
    print("DEPTH GRID WITH INTERPOLATION")
    print("=" * 35)

    # Paths
    csv_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_quadrants.csv"
    output_png = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/depth_grid_with_interpolation.png"
    output_csv = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_with_interpolated_depths.csv"

    try:
        # Create depth grid with interpolation
        grid_generator = DepthGridWithInterpolation(csv_path)

        # Load data
        grid_generator.load_csv_data()

        # Perform interpolation
        interpolated_df = grid_generator.perform_interpolation(method="linear")

        # Create visualization
        grid_generator.create_depth_grid_with_interpolation(output_png)

        # Export results to CSV
        grid_generator.export_interpolated_csv(output_csv)

        print("\\n" + "=" * 35)
        print("DEPTH GRID WITH INTERPOLATION COMPLETE")
        print("=" * 35)
        print("✅ Spatial interpolation performed (linear method)")
        print("✅ Grid visualization created showing known + interpolated depths")
        print("✅ Blue squares = known depths (thick blue borders)")
        print("✅ Green squares = interpolated depths (green borders)")
        print("✅ White squares = no coverage (outside convex hull)")
        print("✅ Complete results exported to CSV")
        print(f"\\n📁 Files created:")
        print(f"   • {output_png}")
        print(f"   • {output_csv}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
