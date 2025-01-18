Project Structure:
This project implements an AI classification model using a fine-tuning approach, designed for deployment on a web server. The project is modularized into several components, each handling a specific part of the workflow. Below is a detailed description of each module:

engine.py:
Contains the core functions for training and validating the model during the training process. This module also includes helper functions to address common issues in Convolutional Neural Network (CNN) models, such as overfitting and underfitting. Features such as early stopping are implemented here to optimize model performance.

data_setup.py:
Responsible for data preprocessing and preparation. This module includes functions to create training and testing data loaders and apply necessary transformations to the data, ensuring it is in the correct format for training.

util.py:
A utility module that provides supporting functions, such as saving and loading trained models, and other general-purpose utilities to streamline the workflow.

train.py:
The main entry point for training the model. This script brings together all components by declaring the model architecture, loss function, optimizer, and learning rate scheduler. It integrates the functions from the sub-modules to create and train the AI model effectively.

load_model.py:
Used for testing the trained model. This module allows users to load the trained model and test it with custom images, ensuring seamless evaluation of the model's performance.

This modular structure ensures a clean, maintainable codebase and facilitates collaboration and scalability for future enhancements.
