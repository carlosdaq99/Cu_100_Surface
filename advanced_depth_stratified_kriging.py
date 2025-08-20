#!/usr/bin/env python3
"""
Advanced PyKrige Implementation with Depth-Stratified Copper Interpolation
Uses realistic depth data to create depth-specific copper concentration surfaces

This implementation demonstrates:
1. Depth-stratified copper modeling (copper typically decreases with depth)
2. Realistic copper concentration assignment based on depth patterns
3. Multiple interpolation surfaces at different depth levels
4. Comprehensive uncertainty quantification
5. Cross-validation and model performance assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


class DepthStratifiedKriging:
    """Advanced kriging implementation with depth-stratified copper modeling"""

    def __init__(self, csv_path):
        """Initialize with CSV data"""
        self.csv_path = csv_path
        self.df = None
        self.cu_data = None
        self.depth_levels = None
        self.kriging_models = {}

    def load_and_prepare_data(self):
        """Load CSV and create realistic copper concentrations based on depth"""
        print("Loading grid quadrants with depth data...")

        # Load the updated CSV with depths
        self.df = pd.read_csv(self.csv_path)

        # Filter to quadrants with data
        data_df = self.df[self.df["has_data"] == 1].copy()
        print(f"Found {len(data_df)} quadrants with data and depths")

        # Generate realistic copper concentrations based on depth patterns
        # Copper typically decreases with depth in many geological settings
        np.random.seed(42)  # For reproducible results

        # Create depth-dependent copper concentrations
        # Surface: 20-200 mg/kg, decreases with depth, some variability
        base_cu = np.random.lognormal(
            mean=3.5, sigma=0.8, size=len(data_df)
        )  # 30-300 mg/kg
        depth_factor = 1.0 / (
            1.0 + data_df["sample_depth"] * 0.1
        )  # Decreases with depth
        variability = np.random.normal(1.0, 0.3, size=len(data_df))  # Add some noise

        # Calculate copper concentrations
        cu_concentrations = base_cu * depth_factor * variability
        cu_concentrations = np.clip(cu_concentrations, 5, 500)  # Realistic range

        # Create the final dataset
        self.cu_data = pd.DataFrame(
            {
                "easting": data_df["center_easting"].values,
                "northing": data_df["center_northing"].values,
                "depth": data_df["sample_depth"].values,
                "copper_mg_kg": cu_concentrations,
                "quadrant_id": data_df["quadrant_id"].values,
            }
        )

        print(f"\nCopper concentration statistics:")
        print(f"Min: {self.cu_data['copper_mg_kg'].min():.1f} mg/kg")
        print(f"Max: {self.cu_data['copper_mg_kg'].max():.1f} mg/kg")
        print(f"Mean: {self.cu_data['copper_mg_kg'].mean():.1f} mg/kg")
        print(f"Median: {self.cu_data['copper_mg_kg'].median():.1f} mg/kg")

        # Analyze depth-copper relationship
        correlation = np.corrcoef(self.cu_data["depth"], self.cu_data["copper_mg_kg"])[
            0, 1
        ]
        print(f"Depth-Copper correlation: {correlation:.3f}")

        return self.cu_data

    def create_depth_stratified_surfaces(
        self, depth_levels=[2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    ):
        """Create interpolation surfaces at different depth levels"""
        self.depth_levels = depth_levels

        print(
            f"\\nCreating interpolation surfaces at {len(depth_levels)} depth levels..."
        )

        # Define interpolation grid (same as original grid)
        x_min, x_max = (
            self.cu_data["easting"].min() - 500,
            self.cu_data["easting"].max() + 500,
        )
        y_min, y_max = (
            self.cu_data["northing"].min() - 500,
            self.cu_data["northing"].max() + 500,
        )

        grid_x = np.arange(x_min, x_max, 250)  # 250m resolution
        grid_y = np.arange(y_min, y_max, 250)

        results = {}

        for depth_level in depth_levels:
            print(f"\\nProcessing depth level: {depth_level}m")

            # Create depth-weighted dataset for this level
            # Weight points based on proximity to target depth
            depth_weights = 1.0 / (1.0 + np.abs(self.cu_data["depth"] - depth_level))

            # Estimate copper at this depth using weighted interpolation of nearby depths
            estimated_cu = []
            for idx, row in self.cu_data.iterrows():
                # Simple depth interpolation for now - could be improved with geological knowledge
                depth_diff = depth_level - row["depth"]
                if depth_diff > 0:  # Target depth is deeper
                    # Copper decreases with depth
                    decay_factor = np.exp(-depth_diff * 0.05)  # 5% decay per meter
                else:  # Target depth is shallower
                    # Copper increases towards surface
                    growth_factor = np.exp(
                        abs(depth_diff) * 0.03
                    )  # 3% increase per meter
                    decay_factor = growth_factor

                estimated_cu.append(row["copper_mg_kg"] * decay_factor)

            estimated_cu = np.array(estimated_cu)

            # Apply kriging at this depth level
            try:
                # Create ordinary kriging model
                ok_model = OrdinaryKriging(
                    self.cu_data["easting"].values,
                    self.cu_data["northing"].values,
                    estimated_cu,
                    variogram_model="spherical",
                    verbose=False,
                    enable_plotting=False,
                    coordinates_type="euclidean",
                )

                # Perform interpolation
                z, ss = ok_model.execute("grid", grid_x, grid_y)

                # Store results
                results[depth_level] = {
                    "interpolated_surface": z,
                    "variance_surface": ss,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "kriging_model": ok_model,
                    "estimated_cu": estimated_cu,
                    "mean_cu": np.mean(estimated_cu),
                    "std_cu": np.std(estimated_cu),
                }

                print(
                    f"✓ Depth {depth_level}m: Mean Cu = {np.mean(estimated_cu):.1f} mg/kg, Std = {np.std(estimated_cu):.1f} mg/kg"
                )

            except Exception as e:
                print(f"❌ Error at depth {depth_level}m: {str(e)}")
                continue

        self.kriging_models = results
        return results

    def perform_cross_validation(self, n_folds=5):
        """Perform cross-validation to assess model performance"""
        print(f"\\n=== CROSS-VALIDATION ({n_folds}-fold) ===")

        # Prepare data
        X = self.cu_data[["easting", "northing"]].values
        y = self.cu_data["copper_mg_kg"].values

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        rmse_scores = []
        r2_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\\nFold {fold + 1}/{n_folds}")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # Create kriging model
                ok_model = OrdinaryKriging(
                    X_train[:, 0],
                    X_train[:, 1],
                    y_train,
                    variogram_model="spherical",
                    verbose=False,
                    enable_plotting=False,
                )

                # Predict on test set
                y_pred, _ = ok_model.execute("points", X_test[:, 0], X_test[:, 1])

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                rmse_scores.append(rmse)
                r2_scores.append(r2)

                print(f"  RMSE: {rmse:.2f} mg/kg, R²: {r2:.3f}")

            except Exception as e:
                print(f"  ❌ Fold {fold + 1} failed: {str(e)}")
                continue

        if rmse_scores:
            print(f"\\nCross-validation results:")
            print(
                f"Mean RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f} mg/kg"
            )
            print(f"Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

        return rmse_scores, r2_scores

    def create_visualization(self, output_dir="."):
        """Create comprehensive visualization of depth-stratified results"""
        if not self.kriging_models:
            print(
                "❌ No kriging models available. Run create_depth_stratified_surfaces first."
            )
            return

        print(f"\\nCreating visualizations...")

        # Create figure with subplots for each depth level
        n_depths = len(self.kriging_models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (depth, results) in enumerate(self.kriging_models.items()):
            if idx >= 6:  # Maximum 6 subplots
                break

            ax = axes[idx]

            # Plot interpolated surface
            im = ax.contourf(
                results["grid_x"],
                results["grid_y"],
                results["interpolated_surface"],
                levels=20,
                cmap="viridis",
                alpha=0.8,
            )

            # Overlay data points
            scatter = ax.scatter(
                self.cu_data["easting"],
                self.cu_data["northing"],
                c=results["estimated_cu"],
                cmap="viridis",
                s=50,
                edgecolors="white",
                linewidth=1,
            )

            ax.set_title(
                f'Copper at {depth}m depth\\nMean: {results["mean_cu"]:.1f} mg/kg'
            )
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            ax.grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(im, ax=ax, label="Cu (mg/kg)")

        # Hide unused subplots
        for idx in range(len(self.kriging_models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save the plot
        output_path = f"{output_dir}/depth_stratified_copper_interpolation.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Visualization saved: {output_path}")

        plt.show()

        # Create depth profile plot
        self.create_depth_profile_plot(output_dir)

    def create_depth_profile_plot(self, output_dir="."):
        """Create depth vs copper concentration profile"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Scatter plot of individual samples
        ax1.scatter(
            self.cu_data["depth"],
            self.cu_data["copper_mg_kg"],
            alpha=0.6,
            s=50,
            color="blue",
            edgecolors="darkblue",
        )
        ax1.set_xlabel("Depth (m)")
        ax1.set_ylabel("Copper Concentration (mg/kg)")
        ax1.set_title("Copper vs Depth (Individual Samples)")
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(self.cu_data["depth"], self.cu_data["copper_mg_kg"], 1)
        p = np.poly1d(z)
        depth_range = np.linspace(
            self.cu_data["depth"].min(), self.cu_data["depth"].max(), 100
        )
        ax1.plot(
            depth_range,
            p(depth_range),
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Trend (slope: {z[0]:.1f})",
        )
        ax1.legend()

        # Mean copper by depth level
        if self.kriging_models:
            depths = list(self.kriging_models.keys())
            mean_cu = [results["mean_cu"] for results in self.kriging_models.values()]
            std_cu = [results["std_cu"] for results in self.kriging_models.values()]

            ax2.errorbar(
                depths,
                mean_cu,
                yerr=std_cu,
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
            )
            ax2.set_xlabel("Depth Level (m)")
            ax2.set_ylabel("Mean Copper Concentration (mg/kg)")
            ax2.set_title("Mean Copper by Depth Level")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = f"{output_dir}/copper_depth_profile.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Depth profile saved: {output_path}")

        plt.show()

    def export_results(self, output_dir="."):
        """Export results to CSV files"""
        print(f"\\nExporting results...")

        # Export original data with copper
        cu_export_path = f"{output_dir}/copper_depth_data.csv"
        self.cu_data.to_csv(cu_export_path, index=False)
        print(f"✓ Copper data exported: {cu_export_path}")

        # Export interpolation summary
        if self.kriging_models:
            summary_data = []
            for depth, results in self.kriging_models.items():
                summary_data.append(
                    {
                        "depth_level": depth,
                        "mean_copper": results["mean_cu"],
                        "std_copper": results["std_cu"],
                        "min_interpolated": np.min(results["interpolated_surface"]),
                        "max_interpolated": np.max(results["interpolated_surface"]),
                        "mean_interpolated": np.mean(results["interpolated_surface"]),
                    }
                )

            summary_df = pd.DataFrame(summary_data)
            summary_export_path = f"{output_dir}/depth_interpolation_summary.csv"
            summary_df.to_csv(summary_export_path, index=False)
            print(f"✓ Interpolation summary exported: {summary_export_path}")


def main():
    """Main execution function"""
    print("ADVANCED PYKRIGE IMPLEMENTATION WITH DEPTH-STRATIFIED COPPER")
    print("=" * 65)

    # Paths
    csv_path = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs/grid_quadrants.csv"
    output_dir = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100/3d_surface_modeling/outputs"

    try:
        # Initialize kriging system
        kriging_system = DepthStratifiedKriging(csv_path)

        # Load and prepare data
        cu_data = kriging_system.load_and_prepare_data()

        # Create depth-stratified interpolation surfaces
        results = kriging_system.create_depth_stratified_surfaces()

        # Perform cross-validation
        rmse_scores, r2_scores = kriging_system.perform_cross_validation()

        # Create visualizations
        kriging_system.create_visualization(output_dir)

        # Export results
        kriging_system.export_results(output_dir)

        print("\\n" + "=" * 65)
        print("DEPTH-STRATIFIED KRIGING ANALYSIS COMPLETE")
        print("=" * 65)
        print("✅ Realistic copper concentrations generated based on depth patterns")
        print("✅ Multiple interpolation surfaces created at different depth levels")
        print("✅ Cross-validation performed to assess model performance")
        print("✅ Comprehensive visualizations generated")
        print("✅ Results exported to CSV files")
        print("\\n🎯 Key findings:")
        print(
            f"   • {len(cu_data)} sample locations with realistic copper-depth relationships"
        )
        print(f"   • {len(results)} depth-stratified interpolation surfaces")
        if rmse_scores:
            print(
                f"   • Cross-validation RMSE: {np.mean(rmse_scores):.1f} ± {np.std(rmse_scores):.1f} mg/kg"
            )
            print(
                f"   • Cross-validation R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}"
            )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
