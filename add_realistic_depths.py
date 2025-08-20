#!/usr/bin/env python3
"""
Add realistic depth values to grid quadrants CSV
Adds a 'sample_depth' column with geotechnically realistic depths for quadrants with data
"""

import pandas as pd
import numpy as np

def add_realistic_depths(csv_path):
    """
    Add realistic sample depths to quadrants that have data
    
    Args:
        csv_path (str): Path to the grid_quadrants.csv file
    """
    print("Loading grid quadrants data...")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} quadrants")
    
    # Count quadrants with data
    data_quadrants = df[df['has_data'] == 1]
    print(f"Found {len(data_quadrants)} quadrants with data (has_data = 1)")
    
    # Set random seed for reproducible "realistic" depths
    np.random.seed(42)
    
    # Create sample_depth column initialized to None/NaN
    df['sample_depth'] = np.nan
    
    # Generate realistic depths for quadrants with data
    # Typical geotechnical investigation depths:
    # - Shallow investigations: 2-5m
    # - Standard investigations: 5-15m  
    # - Deep investigations: 15-30m
    # Using weighted distribution favoring common depths
    
    n_data = len(data_quadrants)
    
    # Create realistic depth distribution
    # 40% shallow (2-6m), 50% standard (4-12m), 10% deep (10-25m)
    shallow_count = int(0.4 * n_data)
    standard_count = int(0.5 * n_data)
    deep_count = n_data - shallow_count - standard_count
    
    # Generate depths for each category
    shallow_depths = np.random.uniform(2.0, 6.0, shallow_count)
    standard_depths = np.random.uniform(4.0, 12.0, standard_count)
    deep_depths = np.random.uniform(10.0, 25.0, deep_count)
    
    # Combine all depths and shuffle
    all_depths = np.concatenate([shallow_depths, standard_depths, deep_depths])
    np.random.shuffle(all_depths)
    
    # Round to realistic precision (0.5m increments)
    all_depths = np.round(all_depths * 2) / 2
    
    # Assign depths to quadrants with data
    data_indices = df[df['has_data'] == 1].index
    df.loc[data_indices, 'sample_depth'] = all_depths
    
    print(f"\nDepth statistics for {len(data_quadrants)} quadrants with data:")
    depths = df[df['has_data'] == 1]['sample_depth']
    print(f"Min depth: {depths.min():.1f}m")
    print(f"Max depth: {depths.max():.1f}m")
    print(f"Mean depth: {depths.mean():.1f}m")
    print(f"Median depth: {depths.median():.1f}m")
    
    # Show distribution
    shallow = (depths <= 6).sum()
    standard = ((depths > 6) & (depths <= 12)).sum()
    deep = (depths > 12).sum()
    
    print(f"\nDepth distribution:")
    print(f"Shallow (≤6m): {shallow} quadrants ({shallow/len(depths)*100:.1f}%)")
    print(f"Standard (6-12m): {standard} quadrants ({standard/len(depths)*100:.1f}%)")
    print(f"Deep (>12m): {deep} quadrants ({deep/len(depths)*100:.1f}%)")
    
    # Save updated CSV
    output_path = csv_path.replace('.csv', '_with_depths.csv')  # Create new file first
    df.to_csv(output_path, index=False)
    print(f"\n✓ Updated CSV saved with sample_depth column: {output_path}")
    
    # Then overwrite original
    try:
        df.to_csv(csv_path, index=False)
        print(f"✓ Original file updated: {csv_path}")
    except PermissionError:
        print(f"⚠️ Could not overwrite original file (permission denied): {csv_path}")
        print(f"✓ New file created instead: {output_path}")
    
    # Display first few rows with data
    print(f"\nFirst 10 quadrants with data:")
    sample_data = df[df['has_data'] == 1][['quadrant_id', 'row', 'col', 'has_data', 'center_easting', 'center_northing', 'sample_depth']].head(10)
    print(sample_data.to_string(index=False))
    
    return df

def main():
    """Main execution function"""
    csv_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_quadrants.csv"
    
    print("ADDING REALISTIC DEPTHS TO GRID QUADRANTS")
    print("="*50)
    
    try:
        updated_df = add_realistic_depths(csv_path)
        
        print("\n" + "="*50)
        print("DEPTH ASSIGNMENT COMPLETE")
        print("="*50)
        print("✓ Realistic depths added to all quadrants with data")
        print("✓ Depths range from 2-25m with geotechnically realistic distribution")
        print("✓ CSV file updated with new 'sample_depth' column")
        print("✓ Quadrants without data have NaN depth values")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
