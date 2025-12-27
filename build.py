"""
Build script for GBC Emulator
Compiles the project into a standalone executable using PyInstaller.
"""

import subprocess
import sys
import os
import shutil

def main():
    print("=" * 50)
    print("Building GBC Emulator Executable")
    print("=" * 50)
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Clean previous builds
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            print(f"Cleaning {folder}/...")
            shutil.rmtree(folder)
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=GBCEmulator",
        "--onefile",
        "--windowed",  # No console window
        "--add-data", f"src;src",  # Include src folder
        "--hidden-import=numpy",
        "--hidden-import=pygame",
        "--hidden-import=numba",
        "--hidden-import=requests",
        "--collect-all", "pygame",
        "--collect-all", "numba",
        "main.py"
    ]
    
    print("\nRunning PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        exe_path = os.path.join("dist", "GBCEmulator.exe")
        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print()
            print("=" * 50)
            print("BUILD SUCCESSFUL!")
            print(f"  Executable: {os.path.abspath(exe_path)}")
            print(f"  Size: {size_mb:.1f} MB")
            print("=" * 50)
        else:
            print("Build completed but executable not found.")
    else:
        print()
        print("=" * 50)
        print("BUILD FAILED!")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()

