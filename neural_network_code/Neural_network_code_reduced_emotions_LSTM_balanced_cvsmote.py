from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
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

# One-hot encode labels
encoder = OneHotEncoder(handle_unknown='ignore')
Y = encoder.fit_transform(df['detected_emotion'].values.reshape(-1, 1)).toarray()

# Save encoder
encoder_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/emotion_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)
print(f"Encoder saved to '{encoder_path}'")

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Compute class distribution
class_counts = np.sum(Y_train, axis=0)
minority_class_size = int(np.min(class_counts))

print(f"Class sample counts: {class_counts}")
print(f"Minority class size: {minority_class_size}")

# Generate k-values for SMOTE tuning
def generate_k_grid(n_minority):
    k_values = list(range(1, min(16, n_minority))) + [
        k for k in [
            max(1, int(0.01 * n_minority)),
            max(1, int(0.15 * n_minority)),
            max(1, int(n_minority**0.5))
        ] if k < n_minority  # Ensure k is valid
    ]
    return sorted(set(k_values))  # Remove duplicates

k_values = generate_k_grid(minority_class_size)

best_k, best_score = None, 0
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Find the best k value
for k in k_values:
    print(f"\nTesting SMOTE with k = {k}")
    fold_scores = []
    for train_idx, val_idx in skf.split(X_train, np.argmax(Y_train, axis=1)):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

        # Ensure k_neighbors is valid
        valid_k = min(k, minority_class_size - 1)
        
        smote = SMOTE(sampling_strategy='auto', k_neighbors=valid_k, random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_fold.reshape(X_train_fold.shape[0], -1), Y_train_fold)
        X_train_resampled = X_train_resampled.reshape(-1, max_sequence_len, 50)

        model = Sequential([
            Bidirectional(LSTM(100, input_shape=(max_sequence_len, 50))),
            Dropout(0.2),
            Dense(7, activation='softmax')
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_resampled, Y_train_resampled, epochs=2, batch_size=64, verbose=0)
        
        Y_pred = np.argmax(model.predict(X_val_fold), axis=1)
        Y_true = np.argmax(Y_val_fold, axis=1)
        fold_scores.append(accuracy_score(Y_true, Y_pred))
    
    mean_score = np.mean(fold_scores)
    print(f"Mean Accuracy for k = {k}: {mean_score:.4f}")
    
    if mean_score > best_score:
        best_score, best_k = mean_score, k

print(f"\nBest K selected: {best_k}")

smote = SMOTE(sampling_strategy='auto', k_neighbors=best_k, random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), Y_train)
X_train_resampled = X_train_resampled.reshape(-1, max_sequence_len, 50)

print(f"\nBest K selected: {best_k}")
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
model.fit(X_train_resampled, Y_train_resampled, validation_data=(X_test, Y_test), epochs=100, batch_size=64, callbacks=[early_stopping])
model.save('/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/Emotionbhav_hptuned.h5')
print("Model saved to 'Emotionbhav_hptuned.h5'")
print(f"\nBest K selected: {best_k}")