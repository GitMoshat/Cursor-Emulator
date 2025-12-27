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
            try:
                shutil.rmtree(folder)
            except PermissionError:
                print(f"Warning: Could not clean {folder}/ - file may be in use")
    
    # Remove old spec file
    if os.path.exists("GBCEmulator.spec"):
        os.remove("GBCEmulator.spec")
    
    # PyInstaller command - fixed for numba
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=GBCEmulator",
        "--onefile",
        "--windowed",
        "--add-data", f"src;src",
        # Numba requires ALL submodules
        "--collect-all", "numba",
        "--collect-all", "llvmlite",
        # Other deps
        "--collect-all", "pygame",
        "--hidden-import=numpy",
        "--hidden-import=requests",
        # Exclude heavy unused stuff
        "--exclude-module", "matplotlib",
        "--exclude-module", "scipy", 
        "--exclude-module", "pandas",
        "--exclude-module", "tkinter",
        "--exclude-module", "PIL",
        "--exclude-module", "IPython",
        "--exclude-module", "jupyter",
        "main.py"
    ]
    
    print("\nRunning PyInstaller...")
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
