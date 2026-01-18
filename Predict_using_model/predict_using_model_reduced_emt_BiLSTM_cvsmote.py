from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import os
nltk.download('punkt')

# File paths
model_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/Emotionbhav_hptuned.h5'
encoder_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_balanced_cvsmote/emotion_encoder.pkl'
excel_path = '/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx'
output_excel_path = '/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx'

# Load the trained model
model = load_model(model_path)
print(f"Model loaded from '{model_path}'")

# Load the saved OneHotEncoder
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
print(f"Encoder loaded from '{encoder_path}'")

# Load the input Excel file
df = pd.read_excel(excel_path)
print(f"Input data loaded from '{excel_path}'")

# Preprocess the raw_text column
feel_arr = df['raw_text']
feel_arr = [word_tokenize(sent) for sent in feel_arr]

# Define max_sequence_len (ensure it matches the training configuration)
data_path_matched = '/home/sambeg/reduced_emotions/Labelled_using_algo/matched_first_second.xlsx'
df_matched = pd.read_excel(data_path_matched)

# Tokenize sentences
array_text = df['raw_text'].apply(word_tokenize).tolist()
max_sequence_len = max(len(seq) for seq in array_text)

# Function to pad sequences
def padd(arr):
    for i in range(max_sequence_len - len(arr)):
        arr.append('<pad>')
    return arr[:max_sequence_len]

# Tokenize and pad the input
for i in range(len(feel_arr)):
    feel_arr[i] = padd(feel_arr[i])

# Load the GloVe embeddings
vocab_f = 'glove.6B.50d.txt'
embeddings_index = {}
with open(vocab_f, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Embed the sentences
embedded_feel_arr = []
for each_sentence in feel_arr:
    embedded_feel_arr.append([])
    for word in each_sentence:
        if word.lower() in embeddings_index:
            embedded_feel_arr[-1].append(embeddings_index[word.lower()])
        else:
            embedded_feel_arr[-1].append([0] * 50)

# Convert to NumPy array
X = np.array(embedded_feel_arr)

# Predict emotions
predictions = model.predict(X)
decoded_predictions = encoder.inverse_transform(predictions)

# Add predictions to the DataFrame
df['predicted_emotion_LSTM_CVSMOTE'] = decoded_predictions.flatten()


# Save the predictions to the output Excel file
df.to_excel(output_excel_path, index=False)

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

matching_rows = df[(df['label'].str.lower()==df['predicted_emotion_LSTM_CVSMOTE'].str.lower())]

matched_count= len(matching_rows)

matched_percentage= str(round(matched_count*100/total_size,2)) + "%"

    
accuracy_file='/home/sambeg/Accuracy_and_Hyperparameter_results/prediction_accuracy.xlsx'
model_name= "Emotionbhav_reduced_emotions_LSTM_cvsmote"
save_accuracy(accuracy_file, matched_percentage , model_name)


print(matched_percentage)