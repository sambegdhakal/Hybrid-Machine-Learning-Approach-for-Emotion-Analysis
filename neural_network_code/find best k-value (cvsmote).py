from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
import pickle

nltk.download('punkt')

# Load the dataset
data_path = '/home/sambeg/reduced_emotions/Labelled_using_algo/matched_first_second.xlsx'
df = pd.read_excel(data_path)

# Tokenize sentences
array_text = df['raw_text'].apply(word_tokenize).tolist()
max_sequence_len = max(len(seq) for seq in array_text)

def padd(arr):
    arr.extend(['<pad>'] * (max_sequence_len - len(arr)))
    return arr[:max_sequence_len]

array_text = [padd(sent) for sent in array_text]

# Load GloVe embeddings
vocab_f = 'glove.6B.50d.txt'
embeddings_index = {}
with open(vocab_f, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Convert words to embeddings
embedded_array_text = [[embeddings_index.get(word.lower(), [0] * 50) for word in sentence] for sentence in array_text]
X = np.array(embedded_array_text)

# Load encoder
encoder_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/emotion_encoder.pkl'
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
print(f"Encoder loaded from '{encoder_path}'")

# One-hot encode labels
Y = encoder.transform(df['detected_emotion'].values.reshape(-1, 1)).toarray()

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Apply SMOTE with best_k set to 4
best_k = 4
print(f"\nBest K selected: {best_k}")

smote = SMOTE(sampling_strategy='auto', k_neighbors=best_k, random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), Y_train)
X_train_resampled = X_train_resampled.reshape(-1, max_sequence_len, 50)

# Save best_k value
with open("best_k.txt", "w") as file:
    file.write(f"Best K selected: {best_k}\n")

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)

# Define the BiLSTM Model
class BiLSTMModel:
    def __init__(self):
        self.model = Sequential([
            Bidirectional(LSTM(100, input_shape=(max_sequence_len, 50))),
            Dropout(0.2),
            Dense(7, activation='softmax')
        ])
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, Y, validation_data, epochs, batch_size, callbacks=None):
        self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def evaluate(self, X, Y, batch_size):
        return self.model.evaluate(X, Y, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_path):
        self.model.save(file_path)

# Train and save the model
model = BiLSTMModel()
model.fit(X_train_resampled, Y_train_resampled, validation_data=(X_test, Y_test), epochs=50, batch_size=64, callbacks=[early_stopping])
model.save('/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/Emotionbhav_hptuned.h5')
print("Model saved to 'Emotionbhav_hptuned.h5'")
print(f"\nBest K selected: {best_k}")
