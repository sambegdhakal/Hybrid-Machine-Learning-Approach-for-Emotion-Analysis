# Importing the required libraries
import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Bidirectional
from keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import *
from sklearn.model_selection import cross_val_score 
import nltk
import pandas as pd
import pickle 
nltk.download('punkt')

# Load the dataset
data_path = '/home/sambeg/reduced_emotions/Labelled_using_algo/matched_first_second.xlsx'
df = pd.read_excel(data_path)

# The array_text will store all the sentences
array_text=df['raw_text']

# Each  sentence in array_text is tokenized by the help of work tokenizer.
array_text = [word_tokenize(sent) for sent in array_text]


max_sequence_len = max(len(seq) for seq in array_text)

def padd(arr):
    for i in range(max_sequence_len-len(arr)):
        arr.append('<pad>')
    return arr[:max_sequence_len]
   
# call the padd function for each sentence in array_text
for i in range(len(array_text)):
    array_text[i]=padd(array_text[i])


# Glove vector contains a 50 dimensional vector corresponding 
# to each word in dictionary.
vocab_f = 'glove.6B.50d.txt'
 
# embeddings_index is a dictionary which contains the mapping of
# word with its corresponding 50d vector.
embeddings_index = {}
with open(vocab_f, encoding='utf8') as f:
    for line in f:
        # splitting each line of the glove.6B.50d in a list of 
        # items- in which the first element is the word to be embedded,
        # and from second to the end of line contains the 50d vector.
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
         

# Embedding each word of the array_text
embedded_array_text = []
for each_sentence in array_text:
    embedded_array_text.append([])
    for word in each_sentence:
        if word.lower() in embeddings_index:
            embedded_array_text[-1].append(embeddings_index[word.lower()])
        else:
            # if the word to be embedded is '<padd>' append 0 fifty times
            embedded_array_text[-1].append([0]*50)
             

#Converting x into numpy-array
X=np.array(embedded_array_text)

# Perform one-hot encoding on df[0] i.e emotion
encoder = OneHotEncoder(handle_unknown='ignore')
Y = encoder.fit_transform(np.array(df['detected_emotion']).reshape(-1,1)).toarray()


encoder_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models/emotion_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)
print(f"Encoder saved to '{encoder_path}'")

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
													test_size=0.2,
													random_state=42)

early_stopping = EarlyStopping(monitor='val_accuracy',  # You can also use 'val_accuracy'
                               patience=15, 
                               restore_best_weights=True,
                               verbose=1)

class BiLSTMModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(100, input_shape=(100, 50))))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(7, activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, Y, validation_data, epochs, batch_size, callbacks=None):
        self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def evaluate(self, X, Y, batch_size):
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        self.model.save(file_path)  # Properly indented save method

# Create an instance of the BiLSTMModel class
model = BiLSTMModel()

# Fit the model on the input and target data
model.fit(X_train, Y_train,validation_data=(X_test, Y_test), epochs=100, batch_size=64, callbacks=[early_stopping])

# Save the model
model.save('/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models/Emotionbhav_hptuned.h5')
print("Model saved to 'Emotionbhav_hptuned.h5'")
