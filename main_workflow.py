"""
Main execution script for modular geotechnical mapping system.
Coordinates the three main modules: coordinate transformation, quadrant analysis, and map production.
"""

import os
import logging
from coordinate_transformer import CoordinateTransformer
from quadrant_analyzer import QuadrantAnalyzer
from map_imagery_producer import MapImageryProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function that orchestrates the complete mapping workflow.
    """
    logger.info("Starting modular geotechnical mapping workflow")

    # Define file paths
    input_csv = "../CompiledCu.csv"
    outputs_dir = "outputs"

    # Ensure outputs directory exists
    os.makedirs(outputs_dir, exist_ok=True)

    # Output file paths
    coordinates_output = os.path.join(outputs_dir, "coordinates_processed.csv")
    quadrants_output = os.path.join(outputs_dir, "quadrant_summary.csv")
    map_png_output = os.path.join(outputs_dir, "sesro_satellite_map.png")
    map_html_output = os.path.join(outputs_dir, "sesro_interactive_map.html")
    simple_grid_output = os.path.join(outputs_dir, "simple_grid.png")

    try:
        # Step 1: Coordinate Transformation
        logger.info("=" * 60)
        logger.info("STEP 1: COORDINATE CLEANING AND TRANSFORMATION")
        logger.info("=" * 60)

        transformer = CoordinateTransformer()
        df_coords, bounds = transformer.process_csv(input_csv, coordinates_output)

        logger.info(f"✓ Processed {len(df_coords)} test locations")
        logger.info(f"✓ Site area: {bounds['area_km2']:.1f} km²")
        logger.info(f"✓ Coordinate data saved to: {coordinates_output}")

        # Step 2: Quadrant Analysis
        logger.info("=" * 60)
        logger.info("STEP 2: QUADRANT ANALYSIS AND NUMPY ARRAYS")
        logger.info("=" * 60)

        analyzer = QuadrantAnalyzer(quadrant_size=250)
        df_with_quadrants, occupancy_matrix, quadrant_summary = (
            analyzer.process_quadrants(df_coords, bounds, quadrants_output)
        )

        occupied_count = int(occupancy_matrix.sum())
        total_count = int(occupancy_matrix.size)
        occupancy_rate = (occupied_count / total_count) * 100

        logger.info(f"✓ Grid matrix shape: {occupancy_matrix.shape}")
        logger.info(
            f"✓ Occupied quadrants: {occupied_count}/{total_count} ({occupancy_rate:.1f}%)"
        )
        logger.info(f"✓ Quadrant summary saved to: {quadrants_output}")

        # Step 3: Map Production
        logger.info("=" * 60)
        logger.info("STEP 3: SATELLITE MAP IMAGERY PRODUCTION")
        logger.info("=" * 60)

        producer = MapImageryProducer()
        producer.produce_maps(
            df_with_quadrants,
            occupancy_matrix,
            bounds,
            map_png_output,
            map_html_output,
            quadrant_size=250,
        )

        logger.info(f"✓ PNG map generated: {map_png_output}")
        logger.info(f"✓ HTML map generated: {map_html_output}")

        # Step 4: Simple Grid Generation
        logger.info("=" * 60)
        logger.info("STEP 4: SIMPLE GRID GENERATION")
        logger.info("=" * 60)

        from simple_grid_generator import SimpleGridGenerator

        grid_generator = SimpleGridGenerator()
        grid_generator.create_simple_grid(
            occupancy_matrix, bounds, simple_grid_output, quadrant_size=250
        )

        logger.info(f"✓ Simple grid generated: {simple_grid_output}")

        # Final Summary
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        print("\n" + "=" * 60)
        print("MODULAR GEOTECHNICAL MAPPING - EXECUTION SUMMARY")
        print("=" * 60)
        print(f"📊 Total test locations processed: {len(df_coords):,}")
        print(f"🗺️  Site coverage area: {bounds['area_km2']:.1f} km²")
        print(f"⏹️  Grid quadrants (250m×250m): {total_count:,}")
        print(f"✅ Quadrants with data: {occupied_count} ({occupancy_rate:.1f}%)")
        print(f"❌ Empty quadrants: {total_count - occupied_count}")
        print("\n📁 Output Files Generated:")
        print(f"   • {coordinates_output}")
        print(f"   • {quadrants_output}")
        print(f"   • {map_png_output}")
        print(f"   • {map_html_output}")
        print(f"   • {simple_grid_output}")
        print("\n🎯 All modules executed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()
