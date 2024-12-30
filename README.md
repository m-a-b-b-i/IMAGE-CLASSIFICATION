# IMAGE-CLASSIFICATION

MNIST Image Classification
This project demonstrates a simple implementation of an image classification pipeline using the MNIST dataset. The pipeline includes preprocessing, training a neural network, and evaluating the model's performance.

Folder Structure
mnist_classification/
│
├── data/                # Folder to store MNIST data
├── src/                 # Source code
│   ├── __init__.py
│   ├── preprocess.py    # For data preprocessing
│   ├── model.py         # Neural network model
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│
├── requirements.txt     # Python dependencies
└── main.py              # Main script

Prerequisites
Before running the project, ensure you have the following dependencies installed:
Python 3.7+
NumPy
Matplotlib
Scikit-learn

Install Dependencies
To install all required Python libraries, run:
pip install -r requirements.txt

How to Run the Project
Clone the repository or download the project folder.
Navigate to the project directory:

cd mnist_classification
Run the main script:
python main.py


Description
1.Dataset: The project uses the MNIST dataset of handwritten digits.
2.Pipeline:
   -->Preprocessing:
      Load the dataset.
      Normalize the pixel values for consistent scaling.
      Split the dataset into training and testing sets.
   -->Model:
      A simple Multi-Layer Perceptron (MLP) is implemented using Scikit-learn.
      Architecture: Input layer → Hidden layers (64, 32 neurons) → Output layer.
   -->Training:
      The MLP model is trained using the training dataset.
      
Evaluation:
Accuracy and a classification report (precision, recall, F1-score) are computed using the testing dataset.

Project Files
src/preprocess.py: Handles loading, normalization, and splitting of the MNIST dataset.
src/model.py: Defines the MLP neural network model.
src/train.py: Contains the training logic.
src/evaluate.py: Contains evaluation metrics and reporting.
main.py: Combines all the modules to execute the complete pipeline.

Expected Output
When you run the project, you should see:

The model's accuracy on the test dataset.
A detailed classification report with precision, recall, and F1-score for each digit class.
Example Output:
Model Accuracy: 96.78%
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98       980
           1       0.98      0.98      0.98      1135
           2       0.95      0.95      0.95      1032
           ...
           9       0.95      0.96      0.96       980

    accuracy                           0.96     10000
   macro avg       0.96      0.96      0.96     10000
weighted avg       0.96      0.96      0.96     10000

Customization
**Model Architecture: Modify the hidden_layer_sizes in src/model.py to experiment with different network structures.
**Dataset Splitting: Adjust the test_size parameter in src/preprocess.py to change the train-test ratio.
**Hyperparameters: Edit learning rate, activation function, or other parameters in src/model.py.
