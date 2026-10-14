
# Handwritten Digit Recognizer

A deep learning project that recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Features

- Recognizes handwritten digits from uploaded images.
- Uses a CNN trained on the MNIST dataset.
- Displays the predicted digit and model confidence.
- Shows prediction probabilities for all ten digits.
- Provides an interactive web interface using Streamlit.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit

## Model Performance

- Dataset: MNIST
- Training images: 60,000
- Testing images: 10,000
- Training epochs: 5
- Test accuracy: 98.87%

Accuracy is based on the model's evaluation on the MNIST test dataset.

## Project Structure

```text
handwritten-digit-recognizer/
├── app.py
├── train_model.py
├── predict_digit.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── digit_recognizer.keras
```

## Installation

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python train_model.py
```

5. Start the web application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## How to Use

1. Open the Streamlit application.
2. Upload an image of a handwritten digit.
3. Click **Recognize Digit**.
4. View the predicted digit, confidence, and probability chart.

## Future Improvements

- Add a drawing canvas for handwritten digits.
- Improve image preprocessing for different backgrounds.
- Test the application on more handwritten digit images.

## Author

Abhishek Jaiswal
