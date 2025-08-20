"""
SESRO 3D Surface Modeling - Package Installation Script
======================================================

This script implements the immediate Phase 1 package installation strategy
to address critical data quality issues and establish geospatial infrastructure.

Author: Geotechnical Analysis Team
Date: Generated automatically
Purpose: Critical data quality pipeline implementation
"""

import subprocess
import sys
import importlib
import os
from datetime import datetime


def log_message(message, level="INFO"):
    """Log installation progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")


def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name, description=""):
    """Install a package using pip with error handling"""
    log_message(f"Installing {package_name} - {description}")

    try:
        # Check if already installed
        if check_package_installed(package_name):
            log_message(f"{package_name} already installed - skipping", "INFO")
            return True

        # Install package
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            check=True,
        )

        log_message(f"Successfully installed {package_name}", "SUCCESS")
        return True

    except subprocess.CalledProcessError as e:
        log_message(f"Failed to install {package_name}: {e.stderr}", "ERROR")
        return False
    except Exception as e:
        log_message(f"Unexpected error installing {package_name}: {str(e)}", "ERROR")
        return False


def main():
    """Main installation workflow"""
    log_message("Starting SESRO 3D Surface Modeling Package Installation", "INFO")
    log_message("Phase 1: Essential Geospatial Infrastructure", "INFO")

    # Track installation results
    installation_results = {}

    # Phase 1 - Essential Packages (IMMEDIATE PRIORITY)
    essential_packages = [
        ("geopandas", "Coordinate system validation and spatial operations"),
        ("shapely", "Geometric operations and spatial analysis"),
        ("PyKrige", "Professional spatial interpolation (kriging)"),
        ("scikit-gstat", "Geostatistical analysis and variogram modeling"),
        ("scipy", "Scientific computing and spatial algorithms"),
        ("plotly", "Interactive 3D visualization and dashboards"),
        ("matplotlib", "Static publication-quality visualization"),
        ("pandas", "Data processing and analysis"),
        ("numpy", "Numerical computing foundation"),
    ]

    log_message("Installing Tier 1 - Core Geospatial Infrastructure", "INFO")
    log_message("=" * 60)

    for package_name, description in essential_packages:
        success = install_package(package_name, description)
        installation_results[package_name] = success

    # Installation Summary
    log_message("=" * 60)
    log_message("INSTALLATION SUMMARY", "INFO")
    log_message("=" * 60)

    successful_installs = []
    failed_installs = []

    for package, success in installation_results.items():
        if success:
            successful_installs.append(package)
            log_message(f"✅ {package}", "SUCCESS")
        else:
            failed_installs.append(package)
            log_message(f"❌ {package}", "ERROR")

    # Summary statistics
    total_packages = len(installation_results)
    success_count = len(successful_installs)
    failure_count = len(failed_installs)

    log_message(f"Total packages: {total_packages}", "INFO")
    log_message(f"Successful: {success_count}", "SUCCESS")
    log_message(f"Failed: {failure_count}", "ERROR")

    if failure_count > 0:
        log_message("FAILED PACKAGES REQUIRE MANUAL INSTALLATION:", "WARNING")
        for package in failed_installs:
            log_message(f"  pip install {package}", "WARNING")

    # Next steps guidance
    log_message("=" * 60)
    log_message("NEXT STEPS", "INFO")
    log_message("=" * 60)

    if failure_count == 0:
        log_message("🎉 All essential packages installed successfully!", "SUCCESS")
        log_message(
            "Ready to proceed with data quality pipeline implementation", "INFO"
        )
        log_message(
            "Next: Run data validation script to address coordinate issues", "INFO"
        )
    else:
        log_message("⚠️  Some packages failed to install", "WARNING")
        log_message("Manual installation required before proceeding", "WARNING")
        log_message("Resolve installation issues then re-run this script", "INFO")

    # Phase 2 preparation
    log_message("=" * 60)
    log_message("PHASE 2 PACKAGES (Future Installation)", "INFO")
    log_message("When Phase 1 complete, install advanced capabilities:", "INFO")
    log_message("  pip install mayavi vtk open3d gstools", "INFO")
    log_message("=" * 60)

    return failure_count == 0


if __name__ == "__main__":
    try:
        success = main()
        if success:
            log_message("Package installation completed successfully", "SUCCESS")
            sys.exit(0)
        else:
            log_message("Package installation completed with errors", "ERROR")
            sys.exit(1)
    except KeyboardInterrupt:
        log_message("Installation interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log_message(f"Unexpected error during installation: {str(e)}", "ERROR")
        sys.exit(1)
