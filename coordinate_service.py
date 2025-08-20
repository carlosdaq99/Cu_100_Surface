"""
Vectorized Coordinate Service
============================

Fast coordinate transformations based on Dash application patterns.
Key optimization: batch processing instead of loop-based transformations.
"""

import numpy as np
import pandas as pd
import pyproj
from typing import Tuple, Union, List
import logging

logger = logging.getLogger(__name__)


class CoordinateService:
    """
    Efficient coordinate transformation service using vectorized operations.
    Based on Dash application optimization patterns.
    """

    def __init__(self):
        """Initialize coordinate systems once for reuse"""
        self.bng = pyproj.CRS("EPSG:27700")  # British National Grid
        self.wgs84 = pyproj.CRS("EPSG:4326")  # WGS84 (lat/lon)
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(
            self.bng, self.wgs84, always_xy=True
        )
        self.transformer_to_bng = pyproj.Transformer.from_crs(
            self.wgs84, self.bng, always_xy=True
        )

    def bng_to_wgs84_vectorized(
        self,
        eastings: Union[np.ndarray, List[float], float],
        northings: Union[np.ndarray, List[float], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized BNG to WGS84 transformation - processes arrays efficiently.

        Args:
            eastings: Array or list of BNG easting coordinates
            northings: Array or list of BNG northing coordinates

        Returns:
            tuple: (latitudes, longitudes) as numpy arrays
        """
        try:
            # Convert to numpy arrays for vectorized operations
            eastings = np.asarray(eastings)
            northings = np.asarray(northings)

            # Single vectorized transformation call
            lons, lats = self.transformer_to_wgs84.transform(eastings, northings)

            return np.asarray(lats), np.asarray(lons)

        except Exception as e:
            logger.error(f"Vectorized coordinate transformation failed: {e}")
            # Return arrays of NaN with same shape as input
            return np.full_like(eastings, np.nan), np.full_like(northings, np.nan)

    def bng_to_wgs84(self, easting: float, northing: float) -> Tuple[float, float]:
        """
        Single coordinate transformation (backward compatibility).
        For bulk operations, use bng_to_wgs84_vectorized instead.
        """
        lats, lons = self.bng_to_wgs84_vectorized([easting], [northing])
        return float(lats[0]), float(lons[0])

    def wgs84_to_bng_vectorized(
        self,
        latitudes: Union[np.ndarray, List[float], float],
        longitudes: Union[np.ndarray, List[float], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized WGS84 to BNG transformation.

        Args:
            latitudes: Array or list of WGS84 latitude coordinates
            longitudes: Array or list of WGS84 longitude coordinates

        Returns:
            tuple: (eastings, northings) as numpy arrays
        """
        try:
            # Convert to numpy arrays for vectorized operations
            latitudes = np.asarray(latitudes)
            longitudes = np.asarray(longitudes)

            # Single vectorized transformation call
            eastings, northings = self.transformer_to_bng.transform(
                longitudes, latitudes
            )

            return np.asarray(eastings), np.asarray(northings)

        except Exception as e:
            logger.error(f"Vectorized coordinate transformation failed: {e}")
            # Return arrays of NaN with same shape as input
            return np.full_like(latitudes, np.nan), np.full_like(longitudes, np.nan)


# Global instance for reuse
_coordinate_service = None


def get_coordinate_service() -> CoordinateService:
    """
    Get singleton coordinate service instance.
    Pattern used in Dash applications for efficiency.
    """
    global _coordinate_service
    if _coordinate_service is None:
        _coordinate_service = CoordinateService()
    return _coordinate_service


def transform_dataframe_coordinates(
    df: pd.DataFrame, east_col: str = "Easting", north_col: str = "Northing"
) -> pd.DataFrame:
    """
    Transform coordinates for entire DataFrame using vectorized operations.

    Args:
        df: DataFrame with BNG coordinates
        east_col: Name of easting column
        north_col: Name of northing column

    Returns:
        DataFrame with added 'Latitude' and 'Longitude' columns
    """
    coord_service = get_coordinate_service()

    # Vectorized transformation for all rows at once
    lats, lons = coord_service.bng_to_wgs84_vectorized(
        df[east_col].values, df[north_col].values
    )

    # Add columns to dataframe
    df_copy = df.copy()
    df_copy["Latitude"] = lats
    df_copy["Longitude"] = lons

    return df_copy


# Example usage and testing
if __name__ == "__main__":
    # Test vectorized transformations
    test_eastings = [445000, 446000, 447000]
    test_northings = [192000, 193000, 194000]

    coord_service = get_coordinate_service()

    # Test vectorized transformation
    lats, lons = coord_service.bng_to_wgs84_vectorized(test_eastings, test_northings)

    print("Vectorized Coordinate Transformation Test:")
    for i, (e, n, lat, lon) in enumerate(
        zip(test_eastings, test_northings, lats, lons)
    ):
        print(f"  Point {i+1}: {e}, {n} -> {lat:.6f}, {lon:.6f}")

    # Test DataFrame transformation
    test_df = pd.DataFrame(
        {
            "Easting": test_eastings,
            "Northing": test_northings,
            "TestData": ["A", "B", "C"],
        }
    )

    transformed_df = transform_dataframe_coordinates(test_df)
    print(f"\nDataFrame transformation result:")
    print(transformed_df[["Easting", "Northing", "Latitude", "Longitude"]].head())
