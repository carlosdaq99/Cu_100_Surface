"""
Coordinate Transformer Module
Handles cleaning and transformation of coordinates from the original CSV data.
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Handles coordinate cleaning and transformation from British National Grid to WGS84.
    """

    def __init__(self):
        # Initialize transformer for BNG to WGS84 conversion
        self.transformer = Transformer.from_crs(
            "EPSG:27700", "EPSG:4326", always_xy=True
        )

    def load_and_clean_data(self, csv_path):
        """
        Load CSV data and perform initial cleaning.

        Args:
            csv_path (str): Path to the original CSV file

        Returns:
            pd.DataFrame: Cleaned dataframe with valid coordinates
        """
        logger.info(f"Loading data from {csv_path}")

        # Load the CSV data
        df = pd.read_csv(csv_path)

        # Log initial data size
        logger.info(f"Initial data size: {len(df)} records")

        # Clean coordinate data - remove rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=["Easting", "Northing"])

        # Remove rows with zero coordinates
        df = df[(df["Easting"] != 0) & (df["Northing"] != 0)]

        # Log cleaning results
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        logger.info(f"Removed {removed_count} records with invalid coordinates")
        logger.info(f"Cleaned data size: {cleaned_count} records")

        return df

    def transform_coordinates(self, df):
        """
        Transform British National Grid coordinates to WGS84 lat/lon.

        Args:
            df (pd.DataFrame): DataFrame with Easting and Northing columns

        Returns:
            pd.DataFrame: DataFrame with added Latitude and Longitude columns
        """
        logger.info("Transforming coordinates from BNG to WGS84")

        # Vectorized coordinate transformation
        start_time = pd.Timestamp.now()

        lon, lat = self.transformer.transform(
            df["Easting"].values, df["Northing"].values
        )

        # Add transformed coordinates to dataframe
        df = df.copy()
        df["Longitude"] = lon
        df["Latitude"] = lat

        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Coordinate transformation completed in {duration:.4f} seconds")
        logger.info(f"Transformed {len(df)} coordinate pairs")

        return df

    def get_coordinate_bounds(self, df):
        """
        Calculate the bounding box of the coordinates.

        Args:
            df (pd.DataFrame): DataFrame with coordinate columns

        Returns:
            dict: Dictionary containing coordinate bounds
        """
        bounds = {
            "easting_min": df["Easting"].min(),
            "easting_max": df["Easting"].max(),
            "northing_min": df["Northing"].min(),
            "northing_max": df["Northing"].max(),
            "longitude_min": df["Longitude"].min(),
            "longitude_max": df["Longitude"].max(),
            "latitude_min": df["Latitude"].min(),
            "latitude_max": df["Latitude"].max(),
        }

        # Calculate site dimensions
        width_m = bounds["easting_max"] - bounds["easting_min"]
        height_m = bounds["northing_max"] - bounds["northing_min"]
        area_km2 = (width_m * height_m) / 1000000

        bounds["width_m"] = width_m
        bounds["height_m"] = height_m
        bounds["area_km2"] = area_km2

        logger.info(
            f"Site bounds: {width_m:.1f}m x {height_m:.1f}m ({area_km2:.1f} km²)"
        )

        return bounds

    def process_csv(self, csv_path, output_path=None):
        """
        Complete processing pipeline: load, clean, and transform coordinates.

        Args:
            csv_path (str): Path to input CSV file
            output_path (str, optional): Path to save processed data

        Returns:
            tuple: (processed_dataframe, coordinate_bounds)
        """
        # Load and clean data
        df = self.load_and_clean_data(csv_path)

        # Transform coordinates
        df = self.transform_coordinates(df)

        # Calculate bounds
        bounds = self.get_coordinate_bounds(df)

        # Save processed data if output path provided
        if output_path:
            logger.info(f"Saving processed data to {output_path}")
            df.to_csv(output_path, index=False)

        return df, bounds


if __name__ == "__main__":
    # Example usage
    transformer = CoordinateTransformer()

    # Process the original CSV
    input_csv = "../CompiledCu.csv"
    output_csv = "outputs/coordinates_processed.csv"

    df, bounds = transformer.process_csv(input_csv, output_csv)

    print(f"Processed {len(df)} records")
    print(f"Site area: {bounds['area_km2']:.1f} km²")
