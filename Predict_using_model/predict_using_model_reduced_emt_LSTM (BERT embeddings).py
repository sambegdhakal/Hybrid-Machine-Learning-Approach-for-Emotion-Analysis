from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import nltk
import os

nltk.download('punkt')

# File paths
model_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_BERT/Emotionbhav_hptuned.h5'
encoder_path = '/home/sambeg/reduced_emotions/Neural_network_trained_models_LSTM/Hyperparameter_tuned_models_BERT/emotion_encoder.pkl'
excel_path = '/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx'
output_excel_path = '/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx'
accuracy_file = '/home/sambeg/Accuracy_and_Hyperparameter_results/prediction_accuracy.xlsx'

# Load the trained model
model = load_model(model_path)
print(f"Model loaded from '{model_path}'")

# Load the saved OneHotEncoder
with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)
print(f"Encoder loaded from '{encoder_path}'")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load the input Excel file
df = pd.read_excel(excel_path)
print(f"Input data loaded from '{excel_path}'")

# Function to convert text into BERT embeddings
def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Using CLS token

# Generate BERT embeddings for each sentence
X = np.array([get_bert_embeddings(sent) for sent in df['raw_text']])

# Predict emotions
predictions = model.predict(X)
decoded_predictions = encoder.inverse_transform(predictions)

# Add predictions to DataFrame
df['predicted_emotion_LSTM (BERT embeddings)'] = decoded_predictions.flatten()

# Save predictions to Excel
df.to_excel(output_excel_path, index=False)
print(f"Predictions saved to '{output_excel_path}'")

# Accuracy Calculation
total_size = len(df)
matched_count = sum(df['label'].str.lower() == df['predicted_emotion_LSTM (BERT embeddings)'].str.lower())
matched_percentage = round((matched_count / total_size) * 100, 2)
matched_percentage_str = f"{matched_percentage}%"

# Save accuracy results
def save_accuracy(accuracy_file, matched_percentage, model_name):
    new_row = pd.DataFrame([{'Model': model_name, 'Accuracy_percentage': matched_percentage}])

    if os.path.exists(accuracy_file):
        df_acc = pd.read_excel(accuracy_file)
        df_acc = pd.concat([df_acc, new_row], ignore_index=True)
    else:
        df_acc = new_row

    df_acc.to_excel(accuracy_file, index=False)

model_name = "Emotionbhav_reduced_emotions_LSTM (BERT-embeddings)"
save_accuracy(accuracy_file, matched_percentage, model_name)

print(f"Prediction Accuracy: {matched_percentage_str}")
