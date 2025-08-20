#!/usr/bin/env python3
"""
Simple Depth Grid Generator
Creates a grid visualization showing depth values from CSV data
Based on simple_grid_generator.py but using depth values instead of just green/white
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDepthGrid:
    """
    Creates a simple grid visualization showing depth values from CSV data
    """

    def __init__(self, csv_path):
        """Initialize with CSV data path"""
        self.csv_path = csv_path
        self.df = None
        
    def load_csv_data(self):
        """Load the grid quadrants CSV with depth data"""
        logger.info(f"Loading grid data from: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} quadrants")
        
        # Count quadrants with depth data
        with_depth = self.df[self.df['sample_depth'].notna()]
        logger.info(f"Found {len(with_depth)} quadrants with depth values")
        
        return self.df
    
    def create_simple_depth_grid(self, output_path):
        """
        Create a simple grid showing depth values, similar to simple_grid.png
        
        Args:
            output_path (str): Path to save the output PNG
        """
        logger.info(f"Creating simple depth grid: {output_path}")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_csv_data() first.")
        
        # Get grid dimensions
        num_rows = self.df['row'].max() + 1
        num_cols = self.df['col'].max() + 1
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        
        # Draw grid squares
        for _, row_data in self.df.iterrows():
            row = int(row_data['row'])
            col = int(row_data['col'])
            depth = row_data['sample_depth']
            
            # Determine color and content
            if pd.notna(depth):
                # Has depth data - use light blue background
                color = "lightblue"
                text_color = "black"
                text = f"{depth:.1f}"
                border_color = "blue"
                border_width = 1.0
            else:
                # No data - white background
                color = "white"
                text_color = "gray"
                text = ""
                border_color = "lightgray"
                border_width = 0.5
            
            # Create square
            rect = patches.Rectangle(
                (col, row),
                1, 1,
                linewidth=border_width,
                edgecolor=border_color,
                facecolor=color,
                alpha=1.0,
            )
            ax.add_patch(rect)
            
            # Add depth text if available
            if text:
                ax.text(
                    col + 0.5, row + 0.5,
                    text,
                    ha='center', va='center',
                    fontsize=8,
                    fontweight='bold',
                    color=text_color
                )
        
        # Set axis properties
        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        ax.set_aspect("equal")
        
        # Remove axis labels and ticks for clean appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        
        # Add title
        with_depth_count = self.df['sample_depth'].notna().sum()
        total_count = len(self.df)
        
        title = "SESRO Depth Values Grid\\n"
        title += f"{with_depth_count}/{total_count} quadrants with depth data\\n"
        title += "Light Blue = Depth Value (m), White = No Data"
        
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
        
        logger.info(f"Simple depth grid saved: {output_path}")
        logger.info(f"Grid dimensions: {num_cols} x {num_rows}")
        logger.info(f"Blue squares (with depth): {with_depth_count}")
        logger.info(f"White squares (no data): {total_count - with_depth_count}")


def main():
    """Main execution function"""
    print("SIMPLE DEPTH GRID GENERATOR")
    print("=" * 30)
    
    # Paths
    csv_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_quadrants.csv"
    output_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/simple_depth_grid.png"
    
    try:
        # Create simple depth grid
        grid_generator = SimpleDepthGrid(csv_path)
        grid_generator.load_csv_data()
        grid_generator.create_simple_depth_grid(output_path)
        
        print("\\n" + "=" * 30)
        print("SIMPLE DEPTH GRID COMPLETE")
        print("=" * 30)
        print("✅ Grid created showing depth values from CSV")
        print("✅ Light blue squares show depth values")
        print("✅ White squares show no data")
        print(f"✅ Saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
