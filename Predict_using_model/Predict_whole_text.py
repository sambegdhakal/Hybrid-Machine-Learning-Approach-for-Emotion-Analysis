import pandas as pd
from transformers import pipeline

# Load the data
file_path = "/home/sambeg/reduced_emotions/US_tweets_predicted/US_tweets_final_predicted.xlsx"
df = pd.read_excel(file_path)

# Step 1: Load Twitter RoBERTa pipeline for sentiment analysis
emotion_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Step 2: Counter for processed tweets
processed_count = 0

# Step 3: Function to predict sentiment (positive, negative, neutral) using Twitter RoBERTa
def predict_roberta_emotion(text):
    global processed_count

    # Ensure valid, non-empty text
    if not text or not text.strip():
        return None

    # Make prediction
    predictions = emotion_pipeline(text)

    # Get the label with the highest score
    max_label = max(predictions, key=lambda x: x['score'])['label']

    # Map labels to sentiment categories
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }

    # Increment counter and return mapped label
    if max_label in label_mapping:
        processed_count += 1
        return label_mapping[max_label]
    return None

# Step 4: Apply the prediction function to the 'raw_text' field
df['sentiment_roberta'] = df['raw_text'].apply(predict_roberta_emotion)

# Step 5: Save the updated DataFrame to the same Excel file
df.to_excel(file_path, index=False)

# Step 6: Print the total count of processed tweets
print(f"Total tweets processed by Twitter RoBERTa: {processed_count}")

# Step 7: Convert custom emotion to sentiment
def convert_emotion(emotion):
    if emotion in ['happy']:
        return 'positive'
    elif emotion in ['sad', 'angry', 'annoyed', 'fear']:
        return 'negative'
    elif emotion in ['surprise']:
        return 'neutral'
    else:
        return 'unknown'  # For any emotions not covered

# Step 8: Apply emotion conversion
df['converted_emotion'] = df['predicted_emotion'].apply(convert_emotion)

# Step 9: Calculate comparison percentage
# Filter rows where both values are not None/NaN
comparison_df = df.dropna(subset=['converted_emotion', 'sentiment_roberta'])

# Count matching rows
matching_count = (comparison_df['converted_emotion'] == comparison_df['sentiment_roberta']).sum()

# Calculate percentage of matching emotions
total_valid_rows = len(comparison_df)
if total_valid_rows > 0:
    matching_percentage = (matching_count / total_valid_rows) * 100
else:
    matching_percentage = 0.0

# Print the comparison result
print(f"Matching percentage between converted_emotion and sentiment_roberta: {matching_percentage:.2f}%")
