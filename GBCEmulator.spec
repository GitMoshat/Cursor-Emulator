# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['numpy', 'pygame', 'numba', 'requests', 'json']
hiddenimports += collect_submodules('pygame')
hiddenimports += collect_submodules('numba.core')
hiddenimports += collect_submodules('llvmlite')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pandas', 'PIL', 'tkinter', 'test', 'tests', 'unittest', 'doctest', 'pydoc', 'pdb', 'profile', 'cProfile', 'xml', 'xmlrpc', 'html', 'http.server', 'ftplib', 'smtplib', 'imaplib', 'poplib', 'telnetlib', 'uu', 'bz2', 'lzma', 'curses', 'lib2to3', 'idlelib', 'distutils', 'setuptools', 'pkg_resources', 'IPython', 'jupyter', 'notebook', 'numba.cuda', 'llvmlite.tests'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GBCEmulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
