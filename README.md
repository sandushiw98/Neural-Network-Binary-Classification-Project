
# Cat vs Dog Image Classifier

## Project Overview
This project develops a binary classifier using a Convolutional Neural Network (CNN) to distinguish between images of cats and dogs. The model is built using Keras with TensorFlow as the backend. It demonstrates the setup, training, and prediction stages with real-world image data.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- PIL

## Installation
To set up your environment to run this code, you will need to install the necessary Python packages. You can install them using pip:
```bash
pip install tensorflow keras numpy pillow
```

## Dataset
The dataset consists of images of cats and dogs. For training the model, images should be organized into separate directories for each category:
```
/dataset
    /train
        /cats
        /dogs
    /validation
        /cats
        /dogs
```

## Model Architecture
The CNN model is structured as follows:
- Convolutional Layer (32 filters, 3x3 kernel)
- MaxPooling Layer (2x2)
- Convolutional Layer (32 filters, 3x3 kernel)
- MaxPooling Layer (2x2)
- Flatten Layer
- Dense Layer (128 units, ReLU activation)
- Dense Layer (1 unit, Sigmoid activation)

## Training the Model
To train the model, execute the script `train.py`. This script will use images in the `/dataset/train` directory and perform data augmentation to improve the model's generalizability.

## Making Predictions
Use `predict.py` to classify new images. This script will predict whether an image contains a cat or a dog using the trained model. The expected output is either `cat` or `dog`.

## Usage Example
To train the model:
```bash
python train.py
```

To predict a new image:
```bash
python predict.py /path/to/new/image.jpg
```

## Contributing
Contributions to this project are welcome. You can contribute in several ways:
- Reporting bugs
- Suggesting enhancements
- Sending pull requests for bug fixes or new features

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests to us.

## Authors
- Sandushi Weraduwa

## Acknowledgments
- Thanks to anyone whose code was used
