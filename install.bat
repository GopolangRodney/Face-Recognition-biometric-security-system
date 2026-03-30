@echo off
echo Installing Face Recognition Security System...
echo.
echo Step 1: Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Installation failed. Please check the error messages above.
    echo.
    echo If you have issues with dlib, try:
    echo   pip install cmake
    echo   pip install dlib
    echo   pip install face-recognition
    pause
    exit /b 1
)

echo.
echo Step 2: Verifying installation...
python -c "import cv2, face_recognition, numpy; print('✓ All dependencies installed successfully!')"

if %errorlevel% neq 0 (
    echo ERROR: Verification failed. Please install dependencies manually.
    pause
    exit /b 1
)

echo.
echo ======================================
echo Installation completed successfully!
echo ======================================
echo.
echo To start the application, run:
echo   python face_recognition_app.py
echo.
pause
