#!/usr/bin/env python3
"""
Grid Data Exporter for SESRO Site Analysis

This module exports grid data in multiple formats for analysis and visualization,
enabling spatial pattern analysis and 2D interpolation strategy planning.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from coordinate_transformer import CoordinateTransformer
from quadrant_analyzer import QuadrantAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class GridDataExporter:
    """Export grid data in multiple formats for analysis and visualization."""

    def __init__(self):
        """Initialize the grid data exporter."""
        pass

    def export_grid_data(self, occupancy_matrix, bounds, output_dir="outputs"):
        """
        Export grid data in multiple formats for analysis.

        Args:
            occupancy_matrix (np.ndarray): Binary occupancy matrix
            bounds (dict): Coordinate bounds information
            output_dir (str): Output directory path
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(exist_ok=True)

        # 1. Export as JSON with detailed grid information
        self._export_json_format(occupancy_matrix, bounds, output_dir)

        # 2. Export as CSV with coordinates for each quadrant
        self._export_csv_format(occupancy_matrix, bounds, output_dir)

        # 3. Export as text visualization for console viewing
        self._export_text_visualization(occupancy_matrix, output_dir)

        # 4. Export spatial analysis data
        self._export_spatial_analysis(occupancy_matrix, bounds, output_dir)

        logger.info("Grid data exported in multiple formats")

    def _export_json_format(self, occupancy_matrix, bounds, output_dir):
        """Export grid data as JSON with comprehensive information."""
        num_rows, num_cols = occupancy_matrix.shape

        # Create comprehensive grid data structure
        grid_data = {
            "metadata": {
                "grid_dimensions": {
                    "rows": int(num_rows),
                    "cols": int(num_cols),
                    "total_quadrants": int(num_rows * num_cols),
                },
                "quadrant_size_m": 250,
                "site_bounds": {
                    "min_easting": float(bounds["easting_min"]),
                    "max_easting": float(bounds["easting_max"]),
                    "min_northing": float(bounds["northing_min"]),
                    "max_northing": float(bounds["northing_max"]),
                    "width_m": float(bounds["easting_max"] - bounds["easting_min"]),
                    "height_m": float(bounds["northing_max"] - bounds["northing_min"]),
                },
                "occupancy_statistics": {
                    "occupied_quadrants": int(occupancy_matrix.sum()),
                    "empty_quadrants": int(
                        occupancy_matrix.size - occupancy_matrix.sum()
                    ),
                    "occupancy_rate_percent": float(
                        (occupancy_matrix.sum() / occupancy_matrix.size) * 100
                    ),
                },
            },
            "occupancy_matrix": occupancy_matrix.astype(int).tolist(),
            "quadrant_details": [],
        }

        # Add detailed quadrant information
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate quadrant bounds
                easting_min = bounds["easting_min"] + col * 250
                easting_max = easting_min + 250
                northing_min = bounds["northing_min"] + row * 250
                northing_max = northing_min + 250

                quadrant_info = {
                    "row": int(row),
                    "col": int(col),
                    "has_data": bool(occupancy_matrix[row, col]),
                    "quadrant_id": f"R{row:02d}C{col:02d}",
                    "bounds": {
                        "easting_min": float(easting_min),
                        "easting_max": float(easting_max),
                        "northing_min": float(northing_min),
                        "northing_max": float(northing_max),
                        "center_easting": float(easting_min + 125),
                        "center_northing": float(northing_min + 125),
                    },
                }
                grid_data["quadrant_details"].append(quadrant_info)

        # Save JSON file
        json_path = Path(output_dir) / "grid_data_complete.json"
        with open(json_path, "w") as f:
            json.dump(grid_data, f, indent=2)

        logger.info(f"Grid data exported to JSON: {json_path}")

    def _export_csv_format(self, occupancy_matrix, bounds, output_dir):
        """Export grid data as CSV with quadrant coordinates."""
        num_rows, num_cols = occupancy_matrix.shape

        # Create CSV data
        csv_data = []
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate quadrant bounds
                easting_min = bounds["easting_min"] + col * 250
                northing_min = bounds["northing_min"] + row * 250

                csv_data.append(
                    {
                        "quadrant_id": f"R{row:02d}C{col:02d}",
                        "row": row,
                        "col": col,
                        "has_data": int(occupancy_matrix[row, col]),
                        "easting_min": easting_min,
                        "easting_max": easting_min + 250,
                        "northing_min": northing_min,
                        "northing_max": northing_min + 250,
                        "center_easting": easting_min + 125,
                        "center_northing": northing_min + 125,
                    }
                )

        # Save CSV file
        df = pd.DataFrame(csv_data)
        csv_path = Path(output_dir) / "grid_quadrants.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Grid quadrant data exported to CSV: {csv_path}")

    def _export_text_visualization(self, occupancy_matrix, output_dir):
        """Export text-based visualization of the grid."""
        num_rows, num_cols = occupancy_matrix.shape

        # Create text visualization
        text_viz = []
        text_viz.append("SESRO Site Grid Visualization")
        text_viz.append("=" * 50)
        text_viz.append(f"Grid dimensions: {num_cols} columns × {num_rows} rows")
        text_viz.append("Legend: ■ = Data present, □ = No data")
        text_viz.append("")

        # Add column headers
        col_header = "   "
        for col in range(0, num_cols, 5):
            col_header += f"{col:2d}    "
        text_viz.append(col_header)
        text_viz.append("")

        # Create the grid visualization (row 0 at top)
        for row in range(num_rows - 1, -1, -1):  # Start from top row visually
            row_str = f"{row:2d} "
            for col in range(num_cols):
                if occupancy_matrix[row, col]:
                    row_str += "■"
                else:
                    row_str += "□"
            text_viz.append(row_str)

        text_viz.append("")
        text_viz.append(f"Data quadrants: {int(occupancy_matrix.sum())}")
        text_viz.append(
            f"Empty quadrants: {int(occupancy_matrix.size - occupancy_matrix.sum())}"
        )
        text_viz.append(
            f"Occupancy rate: {(occupancy_matrix.sum() / occupancy_matrix.size * 100):.1f}%"
        )

        # Save text file
        txt_path = Path(output_dir) / "grid_visualization.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_viz))

        logger.info(f"Text visualization exported: {txt_path}")

    def _export_spatial_analysis(self, occupancy_matrix, bounds, output_dir):
        """Export spatial analysis data for interpolation planning."""
        num_rows, num_cols = occupancy_matrix.shape

        # Find all data quadrants
        data_quadrants = []
        for row in range(num_rows):
            for col in range(num_cols):
                if occupancy_matrix[row, col]:
                    easting = bounds["easting_min"] + col * 250 + 125  # Center point
                    northing = bounds["northing_min"] + row * 250 + 125
                    data_quadrants.append(
                        {
                            "row": row,
                            "col": col,
                            "easting": easting,
                            "northing": northing,
                        }
                    )

        # Analyze spatial patterns
        analysis = {
            "data_quadrants": data_quadrants,
            "spatial_analysis": {
                "total_data_quadrants": len(data_quadrants),
                "grid_coverage": {
                    "rows_with_data": len({q["row"] for q in data_quadrants}),
                    "cols_with_data": len({q["col"] for q in data_quadrants}),
                    "max_row_gap": self._find_max_gap(occupancy_matrix, axis=0),
                    "max_col_gap": self._find_max_gap(occupancy_matrix, axis=1),
                },
            },
            "interpolation_considerations": {
                "quadrant_size_m": 250,
                "max_interpolation_distance_recommended": 500,  # 2 quadrant spacing
                "sparse_regions": self._identify_sparse_regions(occupancy_matrix),
            },
        }

        # Save analysis file
        analysis_path = Path(output_dir) / "spatial_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Spatial analysis exported: {analysis_path}")

    def _find_max_gap(self, matrix, axis):
        """Find the maximum gap between data points along an axis."""
        if axis == 0:  # Row-wise gaps
            gaps = []
            for col in range(matrix.shape[1]):
                col_data = matrix[:, col]
                data_positions = np.nonzero(col_data)[0]
                if len(data_positions) > 1:
                    gaps.extend(np.diff(data_positions) - 1)
            return int(max(gaps)) if gaps else 0
        else:  # Column-wise gaps
            gaps = []
            for row in range(matrix.shape[0]):
                row_data = matrix[row, :]
                data_positions = np.nonzero(row_data)[0]
                if len(data_positions) > 1:
                    gaps.extend(np.diff(data_positions) - 1)
            return int(max(gaps)) if gaps else 0

    def _identify_sparse_regions(self, matrix):
        """Identify regions with low data density for interpolation planning."""
        num_rows, num_cols = matrix.shape
        sparse_regions = []

        # Analyze 5x5 regions for data density
        for start_row in range(0, num_rows - 4, 5):
            for start_col in range(0, num_cols - 4, 5):
                region = matrix[start_row : start_row + 5, start_col : start_col + 5]
                density = region.sum() / region.size

                if density < 0.2:  # Less than 20% occupancy
                    sparse_regions.append(
                        {
                            "region_bounds": {
                                "start_row": int(start_row),
                                "end_row": int(min(start_row + 5, num_rows)),
                                "start_col": int(start_col),
                                "end_col": int(min(start_col + 5, num_cols)),
                            },
                            "data_density": float(density),
                            "quadrants_with_data": int(region.sum()),
                        }
                    )

        return sparse_regions


if __name__ == "__main__":
    # Example usage
    from coordinate_transformer import CoordinateTransformer
    from quadrant_analyzer import QuadrantAnalyzer

    # Load and process data
    transformer = CoordinateTransformer()
    df, bounds = transformer.process_csv("../CompiledCu.csv")

    # Analyze quadrants
    analyzer = QuadrantAnalyzer(quadrant_size=250)
    df_with_quadrants, matrix, summary = analyzer.process_quadrants(df, bounds)

    # Export grid data
    exporter = GridDataExporter()
    exporter.export_grid_data(matrix, bounds)

    print("Grid data exported successfully!")
    print("Files created:")
    print("- grid_data_complete.json (comprehensive grid data)")
    print("- grid_quadrants.csv (quadrant coordinates and status)")
    print("- grid_visualization.txt (text-based grid view)")
    print("- spatial_analysis.json (interpolation planning data)")
