# Facial Emotion Recognition using CNN

[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/17arhaan/open-source-badges/)

This is a deep learning project entirely based on neural networks for Facial Emotion Recognition (FER). This project demonstrates one of the classical applications in deep learning and computer vision.

Facial emotions play a vital role in our day-to-day life. This system is capable of recognizing facial emotions and can be used as a foundation for emotion-aware applications.

## Project Overview

I trained a Convolutional Neural Network (CNN) with the Kaggle facial emotion dataset to learn patterns for each facial expression and accurately detect facial emotions in real-time.

## Features

- Real-time facial emotion detection
- Support for multiple emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- Face detection using Haar Cascade classifiers
- CNN-based emotion classification
- Easy-to-use implementation with comprehensive documentation

## Installation

### Python Libraries Required:
- keras with tensorflow as backend
- OpenCV
- numpy
- pandas
- matplotlib

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib
```

## Dataset

This project uses the Kaggle Facial Expression Recognition Challenge dataset:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview

You can use the Kaggle API to access and download the dataset.

## Usage

This project is implemented in Jupyter notebook format. The main components include:

- **Facial_Emotion_Recognition_using_CNN.ipynb** - Main implementation file
- **model.h5** - Saved weights of the trained model (200 epochs)
- **model.json** - Saved architecture of the neural network
- **haarcascade classifiers** - Pre-trained classifiers for face detection

### Running the Project

1. Clone this repository
2. Install the required dependencies
3. Download the dataset from Kaggle
4. Run the Jupyter notebook
5. The model will detect faces and classify emotions in real-time

## Model Architecture

The CNN architecture includes:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for multi-class emotion prediction

## Results

The model successfully detects and classifies facial emotions with high accuracy. The system can:
- Detect multiple faces in a single image
- Classify emotions for each detected face
- Provide real-time emotion recognition

## Contributing

All pull requests are welcome! I appreciate any suggestions or improvements.

[![GitHub issues](https://img.shields.io/github/issues/17arhaan/facial-recognition)](https://github.com/17arhaan/facial-recognition/issues)

## License & Copyright

© 17arhaan, Computer Science  
Licensed under the [MIT License](LICENSE)

[![GitHub license](https://img.shields.io/github/license/17arhaan/facial-recognition)](https://github.com/17arhaan/facial-recognition/blob/master/LICENSE)

## Contact

Feel free to reach out for any questions or collaborations:

- GitHub: [17arhaan](https://github.com/17arhaan)
- Project Repository: [facial-recognition](https://github.com/17arhaan/facial-recognition)

---

**Note**: This project serves as a foundation for emotion-aware applications and can be extended for various use cases in human-computer interaction, sentiment analysis, and AI-powered systems.
  
