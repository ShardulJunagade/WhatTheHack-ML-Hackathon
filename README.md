# What the Hack: Machine Learning Hackathon

This repository contains my submission for the "What The Hack" ML hackathon organized by the Technical Council at IIT Gandhinagar on 1 September, 2024.

- Name: Shardul Junagade
- Batch: BTech CSE '23
- Roll Number: 23110297

## Problem Statement:
**Develop a neural network that learns to apply the Sobel filter to images.**

### Tasks:
1. Dataset Preparation:
    - Use the provided [dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) of images.
    - Apply the [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator) to the dataset using a standard image processing library.
    - Save the original and Sobel-filtered image pairs.
2. Model Development:
Design a neural network that takes an original image as input and produces a Sobel-filtered image as output.
3. Training:
Train your model using the prepared dataset.
4. Evaluation:
Evaluate your model's performance on a provided test set.

**Bonus Tasks:** 
Extend your project by applying the same approach to other classical image filters such as the Laplacian Filter and Prewitt Filter. Additionally, visualize and compare representations of different CNN layers as images. Explore various CNN architectures to gain deeper insights into their effects and performance. This will demonstrate your ability to generalize and analyze the model's behavior across different scenarios.


## Installation and Usage
1. Clone: Clone the repository to your local machine.
    ```bash
    git clone "https://github.com/ShardulJunagade/WhatTheHack-ML-Hackathon.git"
    cd "WhatTheHack-ML-Hackathon"
    ```
2. Install Dependencies: Install the required libraries specified in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Notebook: Start Jupyter Notebook or another compatible environment and open the `sobel-hackathon.ipynb` file for running the code.


#
# Project Overview

This project involves developing a Convolutional Neural Network (CNN) that learns to apply the Sobel filter to images. The Sobel filter is a classical image processing filter used for edge detection. The core aim of this project is to demonstrate how CNNs can learn to approximate such traditional filters. The network is based on the U-Net architecture and is trained to predict the output of a Sobel filter applied to images. The model is implemented in TensorFlow and Keras, and this project includes options for using a learning rate scheduler, model evaluation metrics, and sample visualizations.

The project also explores other classical filters like Laplacian and Prewitt filters in the bonus section.

The complete code can be found in `sobel-hackathon.ipynb` file. All the preprocessing and splitting data into train and test sets has been done in the `preprocessing.ipynb` file.

Link to Kaggle Notebooks: [Base Tasks](https://www.kaggle.com/code/sharduljunagade/sobel-hackathon), [Bonus Tasks](https://www.kaggle.com/code/sharduljunagade/bonus-hackathon)

The project is divided into several key tasks:

1. Dataset Preparation
2. Model Development
3. Training with a learning rate scheduler
4. Training without a learning rate schedueler
5. Evaluating the model

The graphical comparisons between the 2 cases can be found in the `lr_scheduler_comparison.ipynb` file.


## Dataset Preparation
For this task, the dataset consists of grayscale images on which the Sobel filter is applied to create a ground truth for edge detection. The goal is for the model to learn how to apply the Sobel filter purely from training data.


- The raw images are loaded and converted to grayscale (if necessary).
- The Sobel filter is applied using a standard image processing library, OpenCV to create the target outputs.
    ```py
    def apply_sobel_filter(image):
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(sobel)
    ```
- The dataset is split into training, validation, and test sets for evaluation purposes.
- These datasets are saved in the `Saved_Datasets` folder in .npy format.
    - X_train and X_test: Original images
    - y_train and y_test: Ground truth Sobel-filtered images


## Model Architecture
The model is built using the U-Net architecture, which is commonly used for image segmentation tasks. It includes:

- Encoder: Consists of convolutional layers with increasing filter sizes, followed by max-pooling layers.
- Bottleneck: A set of convolutional layers that forms the latent representation.
- Decoder: Consists of transpose convolutional layers that upsample the feature maps, followed by concatenation with skip connections from the encoder.

The final output is a single-channel image representing the predicted Sobel-filtered image.
```py
inputs = layers.Input(shape=(256, 256, 1))
c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
...
outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c5)

```

## Training the model
The model is compiled using the Adam optimizer and trained using Mean Squared Error (MSE) loss. The optional learning rate scheduler reduces the learning rate after 5 epochs to improve convergence.


### Learning Rate Scheduler
An optional learning rate scheduler is implemented to reduce the learning rate after 5 epochs exponentially.
```py
def lr_scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
```

## Evaluation
The model is evaluated using several metrics:

- Mean Squared Error (MSE): Measures the average squared difference between predicted and true values.
- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and true values.
- Structural Similarity Index (SSIM): Measures similarity between images
- Custom Accuracy: A pixel-wise accuracy metric that counts how many pixels in the predicted image are within a specified threshold from the true values.
```py
def custom_accuracy(y_true, y_pred, threshold=0.1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff = tf.abs(y_true - y_pred)
    correct_pixels = tf.less_equal(diff, threshold)
    accuracy = tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    return accuracy
```

## Results

After training for 10 epochs, the model achieved the following performance:

| Model                          | MSE (Test) | MAE (Test) | Custom Accuracy (%) | SSIM (Test) |
|--------------------------------|------------|------------|---------------------|-------------|
| With Learning Rate Scheduler   | 0.0065     | 0.0364     | 92.9              | 0.91        |
| Without Learning Rate Scheduler| 0.0064     | 0.0338    | 93.47               | 0.916        |



## Conclusion

This project demonstrates how a CNN can learn to approximate the Sobel filter, a classical edge detection algorithm. The results show that the network can effectively mimic the Sobel filter's functionality, providing insight into how CNNs can generalize traditional image processing techniques.

## Bonus Tasks
The bonus tasks have been implemented in the `bonus-hackathon.py` file. Refer to `Bonus.md` for more details regarding the bonus tasks.




## Acknowledgments
This project was created as part of a hackathon submission. Thanks to the Technical Council of IITGN (hackathon organizers) for providing me with the opportunity to work on this project.