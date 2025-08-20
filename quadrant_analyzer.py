"""
Quadrant Analyzer Module
Handles placement of tests into 250m² quadrants and creates numpy arrays for analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuadrantAnalyzer:
    """
    Handles spatial analysis by dividing the site into 250m x 250m quadrants
    and tracking which quadrants contain test data.
    """

    def __init__(self, quadrant_size=250):
        """
        Initialize the quadrant analyzer.

        Args:
            quadrant_size (int): Size of each quadrant in meters (default: 250)
        """
        self.quadrant_size = quadrant_size
        self.grid_matrix = None
        self.bounds = None
        self.grid_dims = None

    def calculate_grid_dimensions(self, bounds: Dict) -> Tuple[int, int]:
        """
        Calculate grid dimensions based on coordinate bounds.

        Args:
            bounds (Dict): Dictionary containing coordinate bounds

        Returns:
            Tuple[int, int]: (num_cols, num_rows) for the grid
        """
        width_m = bounds["easting_max"] - bounds["easting_min"]
        height_m = bounds["northing_max"] - bounds["northing_min"]

        # Calculate number of quadrants needed
        num_cols = int(np.ceil(width_m / self.quadrant_size))
        num_rows = int(np.ceil(height_m / self.quadrant_size))

        logger.info(
            f"Grid dimensions: {num_cols} x {num_rows} = {num_cols * num_rows} quadrants"
        )
        logger.info(f"Quadrant size: {self.quadrant_size}m x {self.quadrant_size}m")

        return num_cols, num_rows

    def assign_quadrants(self, df: pd.DataFrame, bounds: Dict) -> pd.DataFrame:
        """
        Assign each test location to its corresponding quadrant.

        Args:
            df (pd.DataFrame): DataFrame with Easting and Northing coordinates
            bounds (Dict): Dictionary containing coordinate bounds

        Returns:
            pd.DataFrame: DataFrame with added quadrant indices
        """
        logger.info("Assigning test locations to quadrants")

        # Store bounds for later use
        self.bounds = bounds

        # Calculate grid dimensions
        num_cols, num_rows = self.calculate_grid_dimensions(bounds)
        self.grid_dims = (num_cols, num_rows)

        # Calculate quadrant indices for each point
        df = df.copy()

        # Normalize coordinates to grid origin
        x_normalized = df["Easting"] - bounds["easting_min"]
        y_normalized = df["Northing"] - bounds["northing_min"]

        # Calculate quadrant indices (0-based)
        df["quadrant_col"] = np.floor(x_normalized / self.quadrant_size).astype(int)
        df["quadrant_row"] = np.floor(y_normalized / self.quadrant_size).astype(int)

        # Ensure indices are within bounds
        df["quadrant_col"] = np.clip(df["quadrant_col"], 0, num_cols - 1)
        df["quadrant_row"] = np.clip(df["quadrant_row"], 0, num_rows - 1)

        # Create unique quadrant ID
        df["quadrant_id"] = df["quadrant_row"] * num_cols + df["quadrant_col"]

        logger.info(f"Assigned {len(df)} test locations to quadrants")

        return df

    def create_occupancy_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create a binary matrix indicating which quadrants contain test data.

        Args:
            df (pd.DataFrame): DataFrame with quadrant assignments

        Returns:
            np.ndarray: Binary matrix (1 = contains tests, 0 = empty)
        """
        logger.info("Creating quadrant occupancy matrix")

        if self.grid_dims is None:
            raise ValueError(
                "Grid dimensions not calculated. Run assign_quadrants first."
            )

        num_cols, num_rows = self.grid_dims

        # Initialize empty matrix
        self.grid_matrix = np.zeros((num_rows, num_cols), dtype=int)

        # Mark quadrants that contain test data
        for _, row in df.iterrows():
            quad_row = int(row["quadrant_row"])
            quad_col = int(row["quadrant_col"])
            self.grid_matrix[quad_row, quad_col] = 1

        # Calculate statistics
        total_quadrants = num_rows * num_cols
        occupied_quadrants = np.sum(self.grid_matrix)
        empty_quadrants = total_quadrants - occupied_quadrants
        occupancy_rate = (occupied_quadrants / total_quadrants) * 100

        logger.info(
            f"Quadrant occupancy: {occupied_quadrants}/{total_quadrants} ({occupancy_rate:.1f}%)"
        )
        logger.info(f"Empty quadrants: {empty_quadrants}")

        return self.grid_matrix

    def get_quadrant_coordinates(self, row: int, col: int) -> Dict:
        """
        Get the coordinate bounds for a specific quadrant.

        Args:
            row (int): Quadrant row index
            col (int): Quadrant column index

        Returns:
            Dict: Dictionary with quadrant coordinate bounds
        """
        if self.bounds is None:
            raise ValueError("Bounds not set. Run assign_quadrants first.")

        # Calculate quadrant bounds
        min_easting = self.bounds["easting_min"] + (col * self.quadrant_size)
        max_easting = min_easting + self.quadrant_size
        min_northing = self.bounds["northing_min"] + (row * self.quadrant_size)
        max_northing = min_northing + self.quadrant_size

        return {
            "min_easting": min_easting,
            "max_easting": max_easting,
            "min_northing": min_northing,
            "max_northing": max_northing,
        }

    def get_quadrant_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a summary of test counts per quadrant.

        Args:
            df (pd.DataFrame): DataFrame with quadrant assignments

        Returns:
            pd.DataFrame: Summary of tests per quadrant
        """
        logger.info("Creating quadrant summary")

        # Group by quadrant and count tests
        # Check if TestType column exists, otherwise skip test type analysis
        if "TestType" in df.columns:
            quadrant_summary = (
                df.groupby(["quadrant_row", "quadrant_col"])
                .agg(
                    {
                        "Easting": "count",  # Count of tests
                        "TestType": lambda x: (
                            x.mode().iloc[0] if len(x.mode()) > 0 else "Mixed"
                        ),  # Most common test type
                    }
                )
                .reset_index()
            )
            quadrant_summary.rename(
                columns={"Easting": "test_count", "TestType": "dominant_test_type"},
                inplace=True,
            )
        else:
            quadrant_summary = (
                df.groupby(["quadrant_row", "quadrant_col"])
                .agg({"Easting": "count"})  # Count of tests
                .reset_index()
            )
            quadrant_summary.rename(columns={"Easting": "test_count"}, inplace=True)
            quadrant_summary["dominant_test_type"] = "Unknown"

        # Add quadrant coordinate bounds
        quadrant_coords = []
        for _, row in quadrant_summary.iterrows():
            coords = self.get_quadrant_coordinates(
                int(row["quadrant_row"]), int(row["quadrant_col"])
            )
            quadrant_coords.append(coords)

        coord_df = pd.DataFrame(quadrant_coords)
        quadrant_summary = pd.concat([quadrant_summary, coord_df], axis=1)

        logger.info(f"Created summary for {len(quadrant_summary)} occupied quadrants")

        return quadrant_summary

    def process_quadrants(
        self, df: pd.DataFrame, bounds: Dict, output_path: str = None
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Complete quadrant analysis pipeline.

        Args:
            df (pd.DataFrame): DataFrame with coordinate data
            bounds (Dict): Coordinate bounds
            output_path (str, optional): Path to save quadrant summary

        Returns:
            Tuple: (df_with_quadrants, occupancy_matrix, quadrant_summary)
        """
        # Assign quadrants to test locations
        df_with_quadrants = self.assign_quadrants(df, bounds)

        # Create occupancy matrix
        occupancy_matrix = self.create_occupancy_matrix(df_with_quadrants)

        # Create quadrant summary
        quadrant_summary = self.get_quadrant_summary(df_with_quadrants)

        # Save quadrant summary if output path provided
        if output_path:
            logger.info(f"Saving quadrant summary to {output_path}")
            quadrant_summary.to_csv(output_path, index=False)

        return df_with_quadrants, occupancy_matrix, quadrant_summary


if __name__ == "__main__":
    # Example usage
    from coordinate_transformer import CoordinateTransformer

    # Load processed coordinates
    transformer = CoordinateTransformer()
    df, bounds = transformer.process_csv("../CompiledCu.csv")

    # Analyze quadrants
    analyzer = QuadrantAnalyzer(quadrant_size=250)
    df_with_quadrants, matrix, summary = analyzer.process_quadrants(
        df, bounds, "outputs/quadrant_summary.csv"
    )

    print(f"Grid matrix shape: {matrix.shape}")
    print(f"Occupied quadrants: {np.sum(matrix)}")
    print(f"Quadrant summary records: {len(summary)}")
