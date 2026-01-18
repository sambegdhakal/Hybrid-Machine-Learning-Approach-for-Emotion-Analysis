import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

# trained model batch4 loaded
loaded_model = tf.keras.models.load_model('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned.h5')

# sample loaded
df = pd.read_excel('/home/sambeg/reduced_emotions/human_labelled/human_labeled.xlsx')
sample_data = df['raw_text']

# word_index loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_wordindex.pkl', 'rb') as file:
    word_index = pickle.load(file)

# max_sequence_len loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_max_sequence_len.pkl', 'rb') as file:
    max_sequence_len = pickle.load(file)

# emotion_label_encoder loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_emotion_label_encoder.pkl', 'rb') as file:
    emotion_label_encoder = pickle.load(file)

# Tokenization
def tokenize(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in sentence.split():
            if word in word_index:
                sequence.append(word_index[word])
        sequences.append(sequence)
    return sequences


sequences = tokenize(sample_data, word_index)
padded_sequences = np.array([np.pad(seq, (0, max_sequence_len - len(seq))) for seq in sequences])


predictions = loaded_model.predict(padded_sequences)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)
predicted_emotions = emotion_label_encoder.inverse_transform(predicted_labels)

predicted_cleaned = []
# Add predictions to dataframe and save
for predicted in predicted_emotions:
    predicted_cleaned.append(predicted.replace("['", "").replace("']", ""))

total_size= df['raw_text'].size


#predicted and cleaned
df['predicted_emotion'] = predicted_cleaned

# save in predicted file
df.to_excel('/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx', index=False)


def save_accuracy(accuracy_file, matched_percentage , model_name):
    # DataFrame row
    new_row = pd.DataFrame([{
    'Model': model_name,
    'Accuracy_percentage': matched_percentage
    }])

    if os.path.exists(accuracy_file):
        df = pd.read_excel(accuracy_file)
        df = pd.concat([df, new_row], ignore_index=True) 
    else:
        df = new_row

    df.to_excel(accuracy_file, index=False)

total_size= df['raw_text'].size

matching_rows = df[(df['label'].str.lower()==df['predicted_emotion'].str.lower())]

matched_count= len(matching_rows)

matched_percentage= str(round(matched_count*100/total_size,2)) + "%"

    
accuracy_file='/home/sambeg/Accuracy_and_Hyperparameter_results/prediction_accuracy.xlsx'
model_name= "Emotionbhav_reduced_emotions_trained_model"
save_accuracy(accuracy_file, matched_percentage , model_name)


print(matched_percentage)


