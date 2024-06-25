Toxic Comment Classification

# Overview

This project aims to classify comments based on their toxicity using a deep learning model. The classification labels include "toxic", "severe toxic", "threat", and "racist". The model is built using multiple LSTM layers to handle sequences of text embeddings effectively.

# Data

The dataset consists of comments and their corresponding labels indicating the type of toxicity. The labels include:

toxic
severe toxic
threat
racist
Example
comment_text	toxic	severe toxic	threat	racist
I hate you	1	0	0	0
I love you	0	0	0	0
You suck	0	1	1	0
Preprocessing

# Tokenization and Sequencing
Comments are tokenized and converted to sequences of integers.
Vocabulary size is limited to 200,000 words.
Maximum sequence length is set to 1500 characters.
Vectorization
The tokenized sequences are vectorized to create a structured format suitable for model input.

# Dataset Preparation

The data is divided into training, validation, and test sets:

Training Set: 70% of the data
Validation Set: 20% of the data
Test Set: 10% of the data
Model Architecture

The model uses the following layers:

Embedding Layer: Converts integer sequences to dense vectors.
Bidirectional LSTM Layer: Captures context from both forward and backward directions.
Fully Connected Layers: Extract features from the LSTM outputs.
Output Layer: Produces a vector of size 6 with sigmoid activation for classification.
Training

The model is trained using binary cross-entropy loss and Adam optimizer. The training process includes validation to monitor the model's performance.

# Usage

Install Dependencies
sh
Copy code
pip install tensorflow
Running the Script
Load the data into a DataFrame with the required structure, preprocess the data, create the dataset, and train the model.

# Evaluation
Evaluate the model on the test set to determine its accuracy and effectiveness in classifying toxic comments.

# Acknowledgements

This project leverages TensorFlow for building and training the deep learning model, utilizing LSTM layers for effective handling of sequential text data.





