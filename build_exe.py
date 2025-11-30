#!/usr/bin/env python3
"""
PyInstaller build script for Nimslo Aligner.

Creates a standalone executable for macOS.

Usage:
    python build_exe.py
    
    # Or with custom options:
    python build_exe.py --onefile --name nimslo-align
"""

import subprocess
import sys
import shutil
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def install_pyinstaller():
    """Install PyInstaller if not present."""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build_executable(
    onefile: bool = True,
    name: str = "nimslo-align",
    clean: bool = True
):
    """
    Build the executable using PyInstaller.
    
    Args:
        onefile: Create single file executable (True) or directory (False)
        name: Name of the output executable
        clean: Clean build directories before building
    """
    code_dir = Path(__file__).parent
    dist_dir = code_dir / "dist"
    build_dir = code_dir / "build"
    spec_file = code_dir / f"{name}.spec"
    
    # Clean previous builds
    if clean:
        print("Cleaning previous builds...")
        for d in [dist_dir, build_dir]:
            if d.exists():
                shutil.rmtree(d)
        if spec_file.exists():
            spec_file.unlink()
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", name,
        "--noconfirm",  # Don't ask for confirmation
    ]
    
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        "nimslo_core",
        "nimslo_core.preprocessing",
        "nimslo_core.segmentation", 
        "nimslo_core.alignment",
        "nimslo_core.gif_generator",
        "PIL",
        "PIL.Image",
        "cv2",
        "numpy",
        "rembg",
        "rembg.session_factory",
        "onnxruntime",
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    
    # Add data files (model weights are downloaded at runtime by rembg)
    # Include the nimslo_core package
    cmd.extend(["--add-data", f"{code_dir / 'nimslo_core'}:nimslo_core"])
    
    # Entry point
    cmd.append(str(code_dir / "nimslo_cli.py"))
    
    print(f"Building executable: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run PyInstaller
    result = subprocess.run(cmd, cwd=str(code_dir))
    
    if result.returncode == 0:
        if onefile:
            exe_path = dist_dir / name
        else:
            exe_path = dist_dir / name / name
        
        print("-" * 50)
        print(f"✓ Build successful!")
        print(f"  Executable: {exe_path}")
        
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        
        print(f"\nTo install system-wide:")
        print(f"  sudo cp {exe_path} /usr/local/bin/{name}")
        
        return True
    else:
        print("✗ Build failed!")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Nimslo Aligner executable")
    parser.add_argument(
        "--onefile",
        action="store_true",
        default=True,
        help="Create single-file executable (default)"
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create directory-based executable"
    )
    parser.add_argument(
        "--name",
        default="nimslo-align",
        help="Name of the executable (default: nimslo-align)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean build directories before building"
    )
    
    args = parser.parse_args()
    
    # Check/install PyInstaller
    if not check_pyinstaller():
        install_pyinstaller()
    
    # Build
    onefile = not args.onedir
    success = build_executable(
        onefile=onefile,
        name=args.name,
        clean=not args.no_clean
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


