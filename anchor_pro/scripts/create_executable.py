import os
import PyInstaller.__main__

# Set environment variables
os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'


# Path to additional project resources
graphics_path = '..\\graphics'

PyInstaller.__main__.run([
    '..\\anchor_pro\\main.py',  # Path to the main Python script you want to convert
    '--onefile',  # Create a one-file bundled executable
    # '--windowed',  # Use this option for GUI applications (remove it for console applications)
    '--hidden-import=openpyxl.cell._writer',
    '--hidden-import=matplotlib.backends.backend_pdf',
    # '--exclude-module=traitsui main.py',
    '--icon=..\\graphics\\DegPyramid.ico',
    f'--add-data={graphics_path};graphics',
    '--name=AnchorPro - v2.2.7'
    # Add any additional options you need here
])
