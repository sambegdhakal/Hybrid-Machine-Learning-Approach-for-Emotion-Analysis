import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import pickle
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs
# Start time
start_time = datetime.now()

# Load the dataset
df = pd.read_excel('/home/sambeg/reduced_emotions/Labelled_using_algo/matched_first_second.xlsx')

# Split the dataset into training (80%) and validation (20%) sets
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

# Process text data to create vocabulary
def process_data(data):
    tokens = set()
    for text in data:
        tokens.update(re.findall(r'\b\w+\b', text.lower()))
    return tokens

train_column_data = train_df['raw_text'].astype(str)
val_column_data = validation_df['raw_text'].astype(str)

# vocabulary and word index created
vocab = process_data(train_column_data)
word_index = {word: index for index, word in enumerate(vocab, start=1)}

# Save vocabulary and word index 
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models_smote/Emotionbhav_hptuned_vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models_smote/Emotionbhav_hptuned_wordindex.pkl', 'wb') as file:
    pickle.dump(word_index, file)

# Tokenizer function to convert text to sequences
def tokenize(sentences):
    sequences = []
    for sentence in sentences:
        sequence = [word_index[word] for word in sentence.split() if word in word_index]
        sequences.append(sequence)
    return sequences

# Tokenize and pad sequences
train_sequences = tokenize(train_column_data)
val_sequences = tokenize(val_column_data)

# Determine the max sequence length using both training and validation data
max_sequence_len = max(
    max(len(seq) for seq in train_sequences),
    max(len(seq) for seq in val_sequences)
)

padded_train_sequences = np.array([np.pad(seq, (0, max_sequence_len - len(seq))) for seq in train_sequences])
padded_val_sequences = np.array([np.pad(seq, (0, max_sequence_len - len(seq))) for seq in val_sequences])

# Save max_sequence_len for later use
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models_smote/Emotionbhav_hptuned_max_sequence_len.pkl', 'wb') as file:
    pickle.dump(max_sequence_len, file)

# Encode and process emotion labels
train_emotion = train_df["detected_emotion"].astype(str).str.lower().str.replace(" ", "")
val_emotion = validation_df["detected_emotion"].astype(str).str.lower().str.replace(" ", "")

cleaned_train_emotion = [emotion.replace("['", "").replace("']", "") for emotion in train_emotion]
cleaned_val_emotion = [emotion.replace("['", "").replace("']", "") for emotion in val_emotion]

# Initialize and fit LabelEncoder
emotion_label_encoder = LabelEncoder()
train_emotion_labels = emotion_label_encoder.fit_transform(cleaned_train_emotion)
val_emotion_labels = emotion_label_encoder.transform(cleaned_val_emotion)

# One-hot encode emotion labels
categorical_train_emotion_labels = np.eye(len(set(train_emotion_labels)))[train_emotion_labels]
categorical_val_emotion_labels = np.eye(len(set(val_emotion_labels)))[val_emotion_labels]

# Save the emotion label encoder for later use
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models_smote/Emotionbhav_hptuned_emotion_label_encoder.pkl', 'wb') as file:
    pickle.dump(emotion_label_encoder, file)

# Balance the dataset using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
padded_train_sequences_resampled, categorical_train_emotion_labels_resampled = smote.fit_resample(
    padded_train_sequences.reshape(padded_train_sequences.shape[0], -1), categorical_train_emotion_labels
)

# Reshape X_train_resampled back to the original shape (number of samples, sequence length)
padded_train_sequences_resampled = padded_train_sequences_resampled.reshape(-1, max_sequence_len)

# Define the model for hyperparameter tuning
def build_tunable_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(vocab) + 1, 
                                        output_dim=hp.Choice('embedding_output_dim', [32, 64, 128]),
                                        input_length=max_sequence_len, 
                                        trainable=True))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=128, step=32), 
                                    activation=hp.Choice('dense_activation', ['relu', 'tanh'])))
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(categorical_train_emotion_labels.shape[1], activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
# Hyperparameter tuning function

def train_model_batchsize(hp):
    batch_size = hp.Int('batch_size', min_value=32, max_value=128, step=8)
    model = build_tunable_model(hp)
    print(f"\nTraining with batch size: {batch_size}\n")
    model.fit(padded_train_sequences_resampled,  
              categorical_train_emotion_labels_resampled,  
              epochs=20,   
              validation_data=(padded_val_sequences, categorical_val_emotion_labels),
              batch_size=batch_size)
    return model

# Set up the tuner for hyperparameter search
tuner = RandomSearch(
    train_model_batchsize,
    objective='val_accuracy',
    max_trials=30,  # Reduced number of trials for efficiency
    executions_per_trial=1,
    directory='directory_hyperparam_tuned_reduced_smote',
    project_name='emotion_classification_tuning_smote'
)

# Perform the search
tuner.search(padded_train_sequences_resampled,  
             categorical_train_emotion_labels_resampled,
             validation_data=(padded_val_sequences, categorical_val_emotion_labels))

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
print(f"Best Hyperparameters: {best_hps.values}")

# Build the model using the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Extract the best batch size for training
best_batch_size = best_hps.get('batch_size') or 64  # Default to 64

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  
    patience=10,  # Reduced patience to speed up training
    restore_best_weights=True,
    verbose=1
)

# Train the final model with the best hyperparameters
history = model.fit(padded_train_sequences_resampled,  
                    categorical_train_emotion_labels_resampled,  
                    epochs=50,  # Reduced to prevent overfitting
                    validation_data=(padded_val_sequences, categorical_val_emotion_labels),
                    batch_size=best_batch_size,
                    callbacks=[early_stopping])

# Evaluate the final model
train_loss, train_accuracy = model.evaluate(padded_train_sequences_resampled, categorical_train_emotion_labels_resampled)
print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

val_loss, val_accuracy = model.evaluate(padded_val_sequences, categorical_val_emotion_labels)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Save the final model
model.save('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models_smote/Emotionbhav_hptuned.h5')
print("Model saved to 'Emotionbhav_hptuned.h5'")

# Record the end time
end_time = datetime.now()
total_time_hours = (end_time - start_time).total_seconds() / 3600
print("Total time in hours:", total_time_hours)

# Save hyperparameters
def save_hyperparameters(hyperparameter_file, best_hps, model_name):
    hps_dictionary = best_hps.values
    hps_dictionary['Model Name'] = model_name

    # DataFrame row
    new_row = pd.DataFrame([hps_dictionary])

    if os.path.exists(hyperparameter_file):
        df = pd.read_excel(hyperparameter_file)
        df = pd.concat([df, new_row], ignore_index=True) 
    else:
        df = new_row

    df.to_excel(hyperparameter_file, index=False)
    
hyperparameter_file = '/home/sambeg/Accuracy_and_Hyperparameter_results/Best_hyperparameters.xlsx'
model_name = "Emotionbhav_reduced_emotions_trained_model_smote"
save_hyperparameters(hyperparameter_file, best_hps, model_name)