# Image Classification with Semi-Supervised Learning

## Introduction

My task was to implement semi-supervised learning to train a model for classifying various classes of plants and dogs. 

### Running the Code

1. **Setup**: 
   - Create an instance on Vertex AI Workbench and use a Jupyter Notebook.
   - Install the Kaggle API to download the dataset. Authentication is required; refer to the [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.
   - Install the following libraries:
     - `torch`
     - `torchvision`
     - `torchaudio`
   - Use version 11.3 of the libraries.
   
2. **Execution**: 
   - After installing the necessary libraries, run all cells in the Jupyter Notebook.
   - The final results will be saved in `submission.csv`.

## Preprocessing

1. **Custom Datasets**:
   - Created custom datasets for both unlabeled and labeled/test images. Labeled images are mapped to their correct labels, while test images are mapped to `None`.
   - The unlabeled dataset returns images with no labels.

2. **Image Transformations**:
   - Applied various transformations from `torchvision.transforms` to the training images, including flipping, rotating, and cropping.
   - Avoided transformations affecting image color to preserve color as an important feature.
   - Combined datasets with and without transformations.
   - All datasets (training labeled, unlabeled, and testing) were resized and normalized.

## Model and Hyperparameters

1. **Model**:
   - Used the ResNet-18 model without pretrained weights to train solely on the provided data.
   - Tested larger models but found ResNet-18 performed best given the small amount of training data.
   - Replaced the fully connected layer with a sequential block consisting of a dropout layer (50% probability) and a linear layer for the final output.

2. **Hyperparameters**:
   - Learning Rate: 0.001
   - Loss Function: Cross entropy loss for labeled training data
   - Optimizer: Adam

## Experimental Setup

1. **Pipeline**:
   - Implemented a training and test loop function.
   - The training loop ran for 25 epochs.
   - During each epoch, images were loaded in batches from the train loader.
   - Applied the model to training images and computed the loss.

2. **Semi-Supervised Learning**:
   - Randomly selected a batch of unlabeled images (batch size of 8).
   - Applied the model to these images and computed probabilities to determine entropy loss.
   - The total loss each iteration included entropy loss (multiplied by a lambda term) and labeled data cross-entropy loss.
   - Lambda started at 0.04 and was increased by 0.04 each epoch.

3. **Data Splitting**:
   - Initially split the training set to use 10% of the data as validation data during model building.
   - For final submission, removed the validation set to maximize the training data.
   - The validation and test sets only underwent basic transformations (resizing to 224x224 and normalizing).
