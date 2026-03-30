# Face Recognition Security System

A Python-based face recognition application that allows you to set a "face password" during enrollment and then verify your identity during authentication.

## Features

- **Face Enrollment**: Capture and store your face as a password
- **Face Authentication**: Verify your identity by showing your face to the camera
- **Visual Feedback**:
  - **Green border** = Face matched successfully (PASSED)
  - **Red border** = Face does not match (FAILED)
- **Security**: 
  - Automatic face detection
  - Prevents multiple faces in one frame
  - Configurable similarity threshold
  - Maximum attempt limit (3 attempts)

## Installation

### 1. Install Python (3.7+)
Download from https://www.python.org/

### 2. Install Required Libraries

Run the following command in your terminal/command prompt:

```bash
pip install -r requirements.txt
```

**Note on dlib**: The `face-recognition` library depends on `dlib`, which requires compilation. If you encounter issues:

**For Windows:**
```bash
pip install cmake
pip install dlib
pip install face-recognition
```


### Run the Application

```bash
python face_recognition_app.py
```

### Menu Options

#### Option 1: Enroll Face (Set Face Password)
1. Select option `1` from the main menu
2. Position your face in the center of the frame
3. Ensure good lighting and clear visibility
4. Press `SPACE` to capture your face
5. The system will detect and store your face encoding
6. You're now registered!

**Tips for best enrollment:**
- Use good lighting (natural light preferred)
- Face the camera directly
- Keep a neutral expression

#### Option 2: Authenticate (Verify Face)
2. Position your face in the camera frame
3. The system automatically scans and compares your face
4. 🟢 **GREEN border + "PASSED"** = Authentication successful!
5. 🔴 **RED border + "FAILED"** = Face does not match
6. You get 3 attempts before lockout

#### Option 3: Exit
Exit the application

## How It Works

### Enrollment Process
1. Camera captures your face
2. Face is detected using HOG (Histogram of Oriented Gradients)
3. Face encoding is created (128-D vector unique to your face)
4. Encoding is saved to `face_encoding.pkl`

### Authentication Process
1. Camera continuously scans for faces
2. Detected face is encoded
3. Encoding is compared with stored face encoding
4. Distance metric determines if it's a match:
   - Distance < 0.6 = **MATCH** (Green border)
   - Distance ≥ 0.6 = **NO MATCH** (Red border)

## File Structure

```
Face Recognition Security System/
├── face_recognition_app.py      # Main application
├── requirements.txt              # Python dependencies
├── README.md                      # This file
└── face_encoding.pkl             # Stored face encoding (created after enrollment)
```

## Troubleshooting

### "No face detected"
- Improve lighting conditions
- Move closer to the camera
- Ensure your entire face is visible
- Remove obstructions (phone, hands, etc.)

### "Multiple faces detected"
- Ensure only you are in the frame
- Ask others to step back

### Camera issues
- Check if camera is properly connected
- Close other applications using the camera
- Restart the application
- Try a different USB port (if external camera)

### Module import errors
```bash
# Reinstall all dependencies
pip install --upgrade --force-reinstall -r requirements.txt
```

### dlib installation problems
- Follow platform-specific instructions in Installation section
- Ensure CMake is installed before dlib
- Check that your Python version is 3.7+

## Security Notes

- Face encoding is stored locally in `face_encoding.pkl`
- No internet connection required
- No data is sent to external servers
- For sensitive applications, consider adding encryption
- Re-enroll if you significantly change appearance (grow beard, cut hair, etc.)

## Performance

- **Enrollment**: ~2-3 seconds per face
- **Authentication**: Real-time (30 FPS on modern hardware)
- **Memory**: ~50-100 MB per face stored
- **CPU**: ~20-30% usage during operation

## Customization

You can modify these parameters in `face_recognition_app.py`:

```python
match_threshold = 0.6      # Lower = stricter matching (range: 0-1)
max_attempts = 3           # Maximum authentication attempts
model = 'hog'             # Or 'cnn' for higher accuracy (slower)
```

## Future Enhancements

- [ ] Multiple face passwords for different users
- [ ] Encryption of stored face encodings
- [ ] Web interface for remote enrollment
- [ ] Liveness detection (prevent spoofing)
- [ ] Integration with system lock/unlock
- [ ] Detailed audit logs
- [ ] Database backend for enterprise use


## Support

For issues or questions, check that:
1. All dependencies are correctly installed
2. Your camera is accessible and working
3. You have Python 3.7 or higher
4. Good lighting conditions exist during enrollment

---

## Gopolang Rodney Diutlwileng
