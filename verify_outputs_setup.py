#!/usr/bin/env python3
"""
Quick verification that the centralized outputs directory is working correctly.
Tests that output paths are properly configured and accessible.
"""

import os
import sys


def main():
    """Verify output directory structure"""
    print("🔍 VERIFYING OUTPUT DIRECTORY CONFIGURATION")
    print("=" * 60)

    # Define paths
    project_root = "c:/Users/dea29431/OneDrive - Rsk Group Limited/Documents/Geotech/SESRO/2025-08-05 Cu LessThan 100"
    outputs_dir = os.path.join(project_root, "3d_surface_modeling", "outputs")

    print(f"📁 Project root: {project_root}")
    print(f"📁 Outputs directory: {outputs_dir}")

    # Check if directories exist
    print(f"\n📋 Directory status:")
    print(f"  Project root exists: {'✓' if os.path.exists(project_root) else '✗'}")
    print(f"  Outputs dir exists: {'✓' if os.path.exists(outputs_dir) else '✗'}")

    # List existing files in outputs directory
    if os.path.exists(outputs_dir):
        files = [
            f
            for f in os.listdir(outputs_dir)
            if os.path.isfile(os.path.join(outputs_dir, f))
        ]
        print(f"\n📂 Files in outputs directory ({len(files)} total):")
        for i, file in enumerate(sorted(files), 1):
            print(f"  {i:2d}. {file}")
    else:
        print(f"\n⚠️  Outputs directory does not exist!")
        return False

    # Check write permissions
    test_file = os.path.join(outputs_dir, "test_write_permissions.tmp")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"\n✓ Write permissions: OK")
    except Exception as e:
        print(f"\n✗ Write permissions: FAILED - {e}")
        return False

    print(f"\n✅ OUTPUT DIRECTORY VERIFICATION COMPLETE")
    print(f"All Python scripts should now save their outputs to:")
    print(f"   {outputs_dir}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
