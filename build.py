"""
Build script for GBC Emulator
Compiles the project into a standalone executable using PyInstaller.
Optimized for smaller size and faster startup.
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
    
    # Exclusions to reduce size - these are not needed at runtime
    excludes = [
        'matplotlib', 'scipy', 'pandas', 'PIL', 'tkinter',
        'test', 'tests', 'unittest', 'doctest',
        'pydoc', 'pdb', 'profile', 'cProfile',
        'xml', 'xmlrpc', 'html', 'http.server',
        'ftplib', 'smtplib', 'imaplib', 'poplib',
        'telnetlib', 'uu', 'bz2', 'lzma',
        'curses', 'lib2to3', 'idlelib',
        'distutils', 'setuptools', 'pkg_resources',
        'IPython', 'jupyter', 'notebook',
        'numba.cuda',  # Don't need CUDA support
        'llvmlite.tests',
    ]
    
    exclude_args = []
    for ex in excludes:
        exclude_args.extend(['--exclude-module', ex])
    
    # PyInstaller command - optimized for size
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=GBCEmulator",
        "--onefile",
        "--windowed",  # No console window
        "--add-data", f"src;src",
        # Core dependencies
        "--hidden-import=numpy",
        "--hidden-import=pygame", 
        "--hidden-import=numba",
        "--hidden-import=requests",
        "--hidden-import=json",
        # Collect only what's needed
        "--collect-submodules", "pygame",
        "--collect-submodules", "numba.core",
        "--collect-submodules", "llvmlite",
        # Optimizations
        "--strip",  # Strip symbols (Linux/Mac)
        "--noupx",  # UPX can cause issues, skip it
        *exclude_args,
        "main.py"
    ]
    
    print("\nRunning PyInstaller (optimized build)...")
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
