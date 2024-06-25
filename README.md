# Comment-Toxicity

Toxic Comment Classification

This project focuses on classifying comments as toxic or not using deep learning. The approach utilizes multiple Long Short-Term Memory (LSTM) layers for handling sequences of embeddings. The labels for the classification include "toxic", "severe toxic", "threat", and "racist".

Data
The data consists of comments and their corresponding labels for different types of toxicity:

toxic
severe toxic
threat
racist
Example:

comment_text	toxic	severe toxic	threat	racist
I hate you	1	0	0	0
I love you	0	0	0	0
You suck	0	1	1	0
Preprocessing
Tokenization and Sequencing

The comments are tokenized and converted to sequences of integers.
Maximum number of features (vocabulary size) is set to 200,000.
Maximum sequence length is set to 1500.
Vectorization

python
Copy code
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1500,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)
Dataset Preparation
The dataset is prepared for training using TensorFlow's data pipeline:

Caching: Speeds up the data retrieval process.
Shuffling: Ensures the data is randomly shuffled.
Batching: Batches of size 16 are created.
Prefetching: Helps prevent data loading bottlenecks.
The dataset is split into:

Training Set: 70% of the data
Validation Set: 20% of the data
Test Set: 10% of the data
python
Copy code
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, Y))
dataset = dataset.cache()
dataset = dataset.shuffle(1600000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))
Model Architecture
The model is built using TensorFlow's Sequential API with the following layers:

Embedding Layer: Converts integer sequences into dense vectors of fixed size.
python
Copy code
model.add(Embedding(MAX_FEATURES + 1, 32))
Bidirectional LSTM Layer: Helps capture the context in both forward and backward directions.
python
Copy code
model.add(Bidirectional(LSTM(32, activation='tanh')))
Fully Connected Layers: For feature extraction.
python
Copy code
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
Output Layer: Outputs a vector of size 6 with sigmoid activation to get values between 0 and 1.
python
Copy code
model.add(Dense(6, activation='sigmoid'))
Training
The model can be trained using:

python
Copy code
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train, validation_data=val, epochs=5)
Usage
Install Dependencies:

sh
Copy code
pip install tensorflow
Run the Script:
Ensure your data is loaded into a DataFrame df with the required structure, then run the script to preprocess the data, create the dataset, and train the model.

Evaluate the Model:
Evaluate the model on the test set to determine its performance.

Acknowledgements
This project uses TensorFlow for building and training the deep learning model. The LSTM layers help in handling the sequential nature of the text data effectively.
