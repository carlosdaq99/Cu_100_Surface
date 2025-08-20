"""
Simple Grid Generator Module
Creates a basic grid visualization with just green/orange squares on white background.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGridGenerator:
    """
    Generates a simple grid visualization with colored squares indicating data presence.
    """

    def __init__(self):
        """Initialize the simple grid generator."""
        pass

    def create_simple_grid(
        self,
        occupancy_matrix: np.ndarray,
        bounds: Dict,
        output_path: str,
        quadrant_size: int = 250,
    ) -> None:
        """
        Create a simple grid visualization with just colored squares.

        Args:
            occupancy_matrix (np.ndarray): Binary matrix of quadrant occupancy
            bounds (Dict): Coordinate bounds dictionary
            output_path (str): Path to save the output PNG
            quadrant_size (int): Size of quadrants in meters
        """
        logger.info(f"Creating simple grid visualization: {output_path}")

        num_rows, num_cols = occupancy_matrix.shape

        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Draw grid squares
        for row in range(num_rows):
            for col in range(num_cols):
                # Determine color based on occupancy
                if occupancy_matrix[row, col] == 1:
                    color = "green"  # Contains data
                else:
                    color = "white"  # Empty - leave white

                # Create square - fix vertical flip by using row directly
                rect = patches.Rectangle(
                    (col, row),  # Use row directly, no flipping
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor=color,
                    alpha=1.0,
                )
                ax.add_patch(rect)

        # Set axis properties
        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        ax.set_aspect("equal")

        # Remove axis labels and ticks for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

        # Add title
        occupied_count = int(occupancy_matrix.sum())
        total_count = int(occupancy_matrix.size)
        occupancy_rate = (occupied_count / total_count) * 100

        title = "SESRO Site Coverage Grid\n"
        title += f"{occupied_count}/{total_count} quadrants with data ({occupancy_rate:.1f}%)\n"
        title += "Green = Data Present, White = No Data"

        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

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

        logger.info(f"Simple grid saved successfully: {output_path}")
        logger.info(f"Grid dimensions: {num_cols} x {num_rows}")
        logger.info(f"Green squares (data): {occupied_count}")
        logger.info(f"White squares (empty): {total_count - occupied_count}")


if __name__ == "__main__":
    # Example usage with existing modules
    from coordinate_transformer import CoordinateTransformer
    from quadrant_analyzer import QuadrantAnalyzer

    # Load and process data
    transformer = CoordinateTransformer()
    df, bounds = transformer.process_csv("../CompiledCu.csv")

    # Analyze quadrants
    analyzer = QuadrantAnalyzer(quadrant_size=250)
    df_with_quadrants, matrix, summary = analyzer.process_quadrants(df, bounds)

    # Generate simple grid
    grid_generator = SimpleGridGenerator()
    grid_generator.create_simple_grid(matrix, bounds, "outputs/simple_grid.png")

    print("Simple grid generated successfully!")
