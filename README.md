# Face Attendance System

An intelligent face recognition-based attendance management system built with deep learning and computer vision technologies.

## Overview

This system utilizes Convolutional Neural Networks (CNN) for accurate face recognition, combined with efficient data management using Pandas and advanced image processing through OpenCV. The application provides real-time face detection and recognition with a user-friendly interface.

## Features

- Deep learning-based face recognition using TensorFlow and Keras
- Real-time face detection with OpenCV Haar Cascades
- Automated attendance tracking with duplicate prevention
- Statistical analysis and reporting
- Natural language processing for attendance logs
- Secure data management with CSV-based storage

## Technology Stack

- **Python 3.8+**
- **TensorFlow 2.13** - Deep learning framework
- **Keras** - Neural network API
- **OpenCV 4.8** - Computer vision library
- **NumPy** - Numerical computing
- **Pandas** - Data analysis and manipulation
- **spaCy** - Natural language processing
- **Tkinter** - GUI framework

## Installation

### Prerequisites

Ensure Python 3.8 or higher is installed on your system.

### Setup

Clone the repository:

```bash
git clone https://github.com/kv-creates/face-attendance-system.git
cd face-attendance-system
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Starting the Application

Run the main application:

```bash
python face_attendance_system.py
```

### Registering Users

1. Click the **REGISTER** button
2. Enter the person's name in the dialog
3. Click **START** to activate the camera
4. The system will capture 30 face samples automatically
5. Wait for the CNN model training to complete
6. Registration is complete when confirmation message appears

### Marking Attendance

1. Click the **ATTENDANCE** button
2. Click **START** to activate the camera
3. Position face in front of the camera
4. The system will automatically recognize and mark attendance
5. Each person can be marked present once per day

### Viewing Statistics

1. Click the **STATISTICS** button
2. Review comprehensive attendance data including:
   - Total attendance days per person
   - First attendance timestamp
   - Overall statistics

## System Architecture

### CNN Model

The face recognition system uses a custom Convolutional Neural Network with the following architecture:

```
Input Layer: 100x100 grayscale images

Convolutional Block 1:
- Conv2D (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)

Convolutional Block 2:
- Conv2D (64 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)

Convolutional Block 3:
- Conv2D (128 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)

Fully Connected Layers:
- Flatten
- Dense (128 units, ReLU activation)
- Dropout (0.5)
- Dense (num_classes, Softmax activation)
```

### Data Pipeline

1. **Image Acquisition**: OpenCV captures frames from camera
2. **Face Detection**: Haar Cascade classifier detects faces
3. **Preprocessing**: Histogram equalization and Gaussian blur
4. **Normalization**: NumPy-based statistical normalization
5. **Recognition**: CNN model predicts identity
6. **Data Storage**: Pandas manages attendance records in CSV format

## Project Structure

```
face-attendance-system/
├── face_attendance_system.py    # Main application
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
│
├── dataset/                      # Face image samples (auto-generated)
├── trainer/                      # Trained model files (auto-generated)
└── attendance/                   # Attendance records (auto-generated)
```

## Configuration

### Camera Settings

Modify the camera source in the code if needed:

```python
self.cap = cv2.VideoCapture(0)  # Change index for different cameras
```

### Recognition Threshold

Adjust the confidence threshold for recognition:

```python
if confidence > 0.7:  # Modify threshold value (0.0 to 1.0)
```

### Training Parameters

Customize the number of face samples:

```python
self.max_captures = 30  # Increase for better accuracy
```

## Data Management

### Attendance Records

Daily attendance is stored in CSV format:

```csv
Name,Time,Date
John Doe,09:15:30,2024-02-07
Jane Smith,09:16:45,2024-02-07
```

### Labels Database

User registration data is maintained in CSV:

```csv
Name,ID,Registration_Date
John Doe,0,2024-02-07 09:00:00
Jane Smith,1,2024-02-07 09:05:00
```

## Performance Optimization

- Early stopping callback prevents overfitting during training
- Dropout layers (0.5) improve model generalization
- Histogram equalization enhances image quality
- Gaussian blur reduces noise in face samples

## Security and Privacy

- Face images are stored locally and not transmitted
- Attendance data is saved in local CSV files
- `.gitignore` prevents sensitive data from being committed
- All data remains on the local system

## Troubleshooting

### Camera Access Issues

If camera fails to open:
- Verify camera is connected and functional
- Try different camera indices (0, 1, 2)
- Check camera permissions in system settings

### Recognition Accuracy

To improve recognition accuracy:
- Ensure adequate lighting during registration
- Capture samples from multiple angles
- Increase number of training samples
- Adjust confidence threshold

### Model Training Errors

If training fails:
- Verify all dependencies are installed
- Check dataset directory contains images
- Ensure sufficient system memory

### spaCy Model Issues

If NLP features fail:
- Reinstall spaCy model: `python -m spacy download en_core_web_sm`
- Verify spaCy installation: `pip install --upgrade spacy`

## Technical Details

### Image Preprocessing

Images undergo multiple preprocessing steps:
- Conversion to grayscale
- Resizing to 100x100 pixels
- Histogram equalization for contrast enhancement
- Gaussian blur for noise reduction
- Statistical normalization using NumPy

### Model Training

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 8
- Epochs: 20 (with early stopping)
- Validation: Early stopping with patience of 5 epochs

### Data Analysis

Pandas provides:
- Grouping by person name
- Aggregation of attendance counts
- Time-series analysis
- CSV import/export operations

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request with detailed description

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Pandas developers for data analysis capabilities
- spaCy team for NLP functionality

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Review existing documentation
- Check troubleshooting section

## Version History

**Version 1.0.0**
- Initial release
- CNN-based face recognition
- Real-time attendance tracking
- Statistical reporting
- NLP integration

## Author

kv-creates

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- OpenCV Documentation: https://docs.opencv.org/
- Pandas Documentation: https://pandas.pydata.org/
- spaCy Documentation: https://spacy.io/
