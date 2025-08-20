"""
Map Imagery Producer Module
Handles creation of satellite imagery maps with test data overlays.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import contextily as ctx
import folium
import logging
from typing import Dict, Tuple, Optional
from pyproj import Transformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MapImageryProducer:
    """
    Produces satellite imagery maps with geotechnical test data overlays.
    """

    def __init__(self):
        """Initialize the map imagery producer."""
        self.transformer = Transformer.from_crs(
            "EPSG:27700", "EPSG:4326", always_xy=True
        )
        self.test_type_colors = {
            "CPTDerived": "#1f77b4",  # Blue
            "HandVane": "#ff7f0e",  # Orange
            "TriaxialTotal": "#2ca02c",  # Green
        }

    def setup_map_figure(
        self, bounds: Dict, figsize: Tuple[int, int] = (15, 12)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Set up the matplotlib figure and axes for map plotting.

        Args:
            bounds (Dict): Coordinate bounds dictionary
            figsize (Tuple[int, int]): Figure size in inches

        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
        """
        logger.info("Setting up map figure")

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Set axis bounds using BNG coordinates
        ax.set_xlim(bounds["easting_min"], bounds["easting_max"])
        ax.set_ylim(bounds["northing_min"], bounds["northing_max"])

        # Configure axis
        ax.set_xlabel("Easting (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Northing (m)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect("equal")

        return fig, ax

    def add_satellite_imagery(
        self, ax: plt.Axes, bounds: Dict, alpha: float = 0.5
    ) -> None:
        """
        Add satellite imagery as background.

        Args:
            ax (plt.Axes): Matplotlib axes object
            bounds (Dict): Coordinate bounds dictionary
            alpha (float): Transparency of satellite imagery
        """
        logger.info("Adding satellite imagery background")

        try:
            # Add satellite imagery using contextily
            ctx.add_basemap(
                ax,
                crs="EPSG:27700",
                source=ctx.providers.Esri.WorldImagery,
                alpha=alpha,
                zoom="auto",
            )
            logger.info("Satellite imagery added successfully")

        except Exception as e:
            logger.warning(f"Failed to add satellite imagery: {e}")
            ax.set_facecolor("lightgray")

    def add_quadrant_grid(
        self,
        ax: plt.Axes,
        occupancy_matrix: np.ndarray,
        bounds: Dict,
        quadrant_size: int = 250,
        occupied_alpha: float = 0.7,
        empty_alpha: float = 0.3,
    ) -> None:
        """
        Add quadrant grid overlay showing data coverage.

        Args:
            ax (plt.Axes): Matplotlib axes object
            occupancy_matrix (np.ndarray): Binary matrix of quadrant occupancy
            bounds (Dict): Coordinate bounds dictionary
            quadrant_size (int): Size of each quadrant in meters
            occupied_alpha (float): Alpha for quadrants with data
            empty_alpha (float): Alpha for empty quadrants
        """
        logger.info("Adding quadrant grid overlay")

        num_rows, num_cols = occupancy_matrix.shape

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate quadrant bounds
                min_easting = bounds["easting_min"] + (col * quadrant_size)
                max_easting = min_easting + quadrant_size
                min_northing = bounds["northing_min"] + (row * quadrant_size)
                max_northing = min_northing + quadrant_size

                # Determine color and alpha based on occupancy
                if occupancy_matrix[row, col] == 1:
                    # Quadrant contains data - green
                    color = "green"
                    alpha = occupied_alpha
                else:
                    # Empty quadrant - orange
                    color = "orange"
                    alpha = empty_alpha

                # Add rectangle patch
                rect = patches.Rectangle(
                    (min_easting, min_northing),
                    quadrant_size,
                    quadrant_size,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor=color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

        logger.info(f"Added {num_rows * num_cols} quadrant overlays")

    def add_test_data_points(
        self, ax: plt.Axes, df: pd.DataFrame, point_size: int = 8, alpha: float = 0.8
    ) -> None:
        """
        Add test data points colored by test type.

        Args:
            ax (plt.Axes): Matplotlib axes object
            df (pd.DataFrame): DataFrame with test location data
            point_size (int): Size of scatter plot points
            alpha (float): Alpha transparency of points
        """
        logger.info("Adding test data points")

        # Check if TestType column exists for coloring
        if "TestType" in df.columns:
            # Group by test type and plot
            for test_type, color in self.test_type_colors.items():
                test_data = df[df["TestType"] == test_type]

                if len(test_data) > 0:
                    ax.scatter(
                        test_data["Easting"],
                        test_data["Northing"],
                        c=color,
                        s=point_size,
                        alpha=alpha,
                        label=f"{test_type} ({len(test_data)})",
                        edgecolor="black",
                        linewidth=0.3,
                    )

                    logger.info(f"Added {len(test_data)} {test_type} points")
        else:
            # Plot all points in one color if no test type column
            ax.scatter(
                df["Easting"],
                df["Northing"],
                c="blue",
                s=point_size,
                alpha=alpha,
                label=f"All Tests ({len(df)})",
                edgecolor="black",
                linewidth=0.3,
            )
            logger.info(f"Added {len(df)} test points (no type classification)")

    def add_map_legend(self, ax: plt.Axes) -> None:
        """
        Add legend for test types and quadrant colors.

        Args:
            ax (plt.Axes): Matplotlib axes object
        """
        logger.info("Adding map legend")

        # Test type legend
        test_legend = ax.legend(
            title="Test Types",
            loc="upper left",
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(0.02, 0.98),
        )
        test_legend.get_frame().set_alpha(0.9)

        # Add quadrant legend manually
        from matplotlib.lines import Line2D

        quadrant_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="green",
                markersize=8,
                alpha=0.7,
                label="Quadrants with Data",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="orange",
                markersize=8,
                alpha=0.3,
                label="Empty Quadrants",
            ),
        ]

        quadrant_legend = ax.legend(
            handles=quadrant_elements,
            title="Quadrant Coverage",
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(0.98, 0.98),
        )
        quadrant_legend.get_frame().set_alpha(0.9)

        # Add the test type legend back
        ax.add_artist(test_legend)

    def add_map_title(self, ax: plt.Axes, df: pd.DataFrame, bounds: Dict) -> None:
        """
        Add descriptive title to the map.

        Args:
            ax (plt.Axes): Matplotlib axes object
            df (pd.DataFrame): DataFrame with test data
            bounds (Dict): Coordinate bounds dictionary
        """
        total_tests = len(df)
        area_km2 = bounds["area_km2"]

        title = f"SESRO Geotechnical Test Locations\n"
        title += f"{total_tests:,} tests across {area_km2:.1f} km² with 250m × 250m quadrant analysis"

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    def produce_map(
        self,
        df: pd.DataFrame,
        occupancy_matrix: np.ndarray,
        bounds: Dict,
        output_path: str,
        quadrant_size: int = 250,
    ) -> None:
        """
        Produce complete satellite imagery map with all overlays.

        Args:
            df (pd.DataFrame): DataFrame with test location data
            occupancy_matrix (np.ndarray): Binary matrix of quadrant occupancy
            bounds (Dict): Coordinate bounds dictionary
            output_path (str): Path to save the output PNG
            quadrant_size (int): Size of quadrants in meters
        """
        logger.info(f"Producing satellite map: {output_path}")

        # Set up figure
        fig, ax = self.setup_map_figure(bounds)

        # Add satellite imagery background
        self.add_satellite_imagery(ax, bounds, alpha=0.5)

        # Add quadrant grid
        self.add_quadrant_grid(ax, occupancy_matrix, bounds, quadrant_size)

        # Add test data points
        self.add_test_data_points(ax, df)

        # Add legends and title
        self.add_map_legend(ax)
        self.add_map_title(ax, df, bounds)

        # Save the map
        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()

        logger.info(f"Map saved successfully: {output_path}")

    def create_interactive_html_map(
        self,
        df: pd.DataFrame,
        occupancy_matrix: np.ndarray,
        bounds: Dict,
        output_path: str,
        quadrant_size: int = 250,
    ) -> None:
        """
        Create an interactive HTML map with toggleable satellite imagery.

        Args:
            df (pd.DataFrame): DataFrame with test location data
            occupancy_matrix (np.ndarray): Binary matrix of quadrant occupancy
            bounds (Dict): Coordinate bounds dictionary
            output_path (str): Path to save the output HTML
            quadrant_size (int): Size of quadrants in meters
        """
        import folium
        from folium import plugins

        logger.info(f"Creating interactive HTML map: {output_path}")

        # Calculate center point in lat/lon
        center_lat = (bounds["latitude_min"] + bounds["latitude_max"]) / 2
        center_lon = (bounds["longitude_min"] + bounds["longitude_max"]) / 2

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles=None,  # We'll add tiles as layers for toggling
        )

        # Add base tile layers that can be toggled
        # OpenStreetMap layer (always visible)
        folium.TileLayer(
            tiles="OpenStreetMap", name="OpenStreetMap", overlay=False, control=True
        ).add_to(m)

        # Satellite imagery layer (toggleable)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite Imagery",
            overlay=False,
            control=True,
        ).add_to(m)

        # Add quadrant grid overlay
        self._add_html_quadrant_grid(m, occupancy_matrix, bounds, quadrant_size)

        # Add test data points
        self._add_html_test_points(m, df)

        # Add layer control for toggling
        folium.LayerControl().add_to(m)

        # Add a legend
        self._add_html_legend(m)

        # Add title and statistics
        total_tests = len(df)
        area_km2 = bounds["area_km2"]
        occupied_count = int(occupancy_matrix.sum())
        total_quadrants = int(occupancy_matrix.size)

        title_html = f"""
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>SESRO Geotechnical Test Locations</h4>
        <p><b>{total_tests:,}</b> tests across <b>{area_km2:.1f} km²</b><br>
        Grid: <b>{occupied_count}/{total_quadrants}</b> quadrants with data 
        ({(occupied_count/total_quadrants*100):.1f}%)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Save the map
        m.save(output_path)
        logger.info(f"Interactive HTML map saved: {output_path}")

    def _add_html_quadrant_grid(
        self, m, occupancy_matrix: np.ndarray, bounds: Dict, quadrant_size: int
    ) -> None:
        """Add quadrant grid to the HTML map."""
        import folium

        num_rows, num_cols = occupancy_matrix.shape

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate quadrant bounds in BNG
                min_easting = bounds["easting_min"] + (col * quadrant_size)
                min_northing = bounds["northing_min"] + (row * quadrant_size)

                # Convert quadrant corners to lat/lon
                corners_lon, corners_lat = self.transformer.transform(
                    [
                        min_easting,
                        min_easting + quadrant_size,
                        min_easting + quadrant_size,
                        min_easting,
                    ],
                    [
                        min_northing,
                        min_northing,
                        min_northing + quadrant_size,
                        min_northing + quadrant_size,
                    ],
                )

                # Create polygon coordinates
                polygon_coords = list(zip(corners_lat, corners_lon))

                # Determine color based on occupancy
                if occupancy_matrix[row, col] == 1:
                    color = "green"
                    fill_opacity = 0.3
                    popup_text = f"Quadrant [{row},{col}] - Contains test data"
                else:
                    color = "orange"
                    fill_opacity = 0.1
                    popup_text = f"Quadrant [{row},{col}] - No test data"

                # Add quadrant rectangle
                folium.Polygon(
                    locations=polygon_coords,
                    color="black",
                    weight=1,
                    fill=True,
                    fillColor=color,
                    fillOpacity=fill_opacity,
                    popup=folium.Popup(popup_text, parse_html=True),
                ).add_to(m)

    def _add_html_test_points(self, m, df: pd.DataFrame) -> None:
        """Add test data points to the HTML map."""
        import folium

        # Define colors for test types
        test_type_colors = {
            "CPTDerived": "#1f77b4",  # Blue
            "HandVane": "#ff7f0e",  # Orange
            "TriaxialTotal": "#2ca02c",  # Green
        }

        # Check if TestType column exists
        if "TestType" in df.columns:
            # Group by test type and add points
            for test_type, color in test_type_colors.items():
                test_data = df[df["TestType"] == test_type]

                if len(test_data) > 0:
                    # Create feature group for this test type
                    feature_group = folium.FeatureGroup(
                        name=f"{test_type} ({len(test_data)} tests)",
                        overlay=True,
                        control=True,
                    )

                    for _, row in test_data.iterrows():
                        popup_text = f"""
                        <b>Test Location</b><br>
                        Type: {test_type}<br>
                        Easting: {row['Easting']:.1f}m<br>
                        Northing: {row['Northing']:.1f}m<br>
                        Location ID: {row.get('LocationID', 'N/A')}
                        """

                        folium.CircleMarker(
                            location=[row["Latitude"], row["Longitude"]],
                            radius=4,
                            popup=folium.Popup(popup_text, parse_html=True),
                            color="black",
                            weight=1,
                            fillColor=color,
                            fillOpacity=0.8,
                        ).add_to(feature_group)

                    feature_group.add_to(m)
        else:
            # Add all points in one color if no test type classification
            feature_group = folium.FeatureGroup(
                name=f"All Tests ({len(df)} tests)", overlay=True, control=True
            )

            for _, row in df.iterrows():
                popup_text = f"""
                <b>Test Location</b><br>
                Easting: {row['Easting']:.1f}m<br>
                Northing: {row['Northing']:.1f}m<br>
                Location ID: {row.get('LocationID', 'N/A')}
                """

                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=4,
                    popup=folium.Popup(popup_text, parse_html=True),
                    color="black",
                    weight=1,
                    fillColor="blue",
                    fillOpacity=0.8,
                ).add_to(feature_group)

            feature_group.add_to(m)

    def _add_html_legend(self, m) -> None:
        """Add a legend to the HTML map."""
        import folium

        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h5>Legend</h5>
        <p><span style="color:green">■</span> Quadrants with Data<br>
           <span style="color:orange">■</span> Empty Quadrants<br>
           <span style="color:#1f77b4">●</span> CPT Tests<br>
           <span style="color:#ff7f0e">●</span> Hand Vane Tests<br>
           <span style="color:#2ca02c">●</span> Triaxial Tests</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    def produce_maps(
        self,
        df: pd.DataFrame,
        occupancy_matrix: np.ndarray,
        bounds: Dict,
        png_output_path: str,
        html_output_path: str,
        quadrant_size: int = 250,
    ) -> None:
        """
        Produce both PNG and interactive HTML maps.

        Args:
            df (pd.DataFrame): DataFrame with test location data
            occupancy_matrix (np.ndarray): Binary matrix of quadrant occupancy
            bounds (Dict): Coordinate bounds dictionary
            png_output_path (str): Path to save the PNG map
            html_output_path (str): Path to save the HTML map
            quadrant_size (int): Size of quadrants in meters
        """
        logger.info("Producing both PNG and HTML maps")

        # Generate PNG map
        self.produce_map(df, occupancy_matrix, bounds, png_output_path, quadrant_size)

        # Generate interactive HTML map
        self.create_interactive_html_map(
            df, occupancy_matrix, bounds, html_output_path, quadrant_size
        )

        logger.info("Both maps generated successfully")


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

    # Produce both PNG and HTML maps
    producer = MapImageryProducer()
    producer.produce_maps(
        df_with_quadrants,
        matrix,
        bounds,
        "outputs/sesro_satellite_map.png",
        "outputs/sesro_interactive_map.html",
    )

    print("Both PNG and HTML maps generated successfully!")
