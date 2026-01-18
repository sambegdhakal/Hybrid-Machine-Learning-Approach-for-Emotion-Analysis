import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the Twitter RoBERTa emotion classification pipeline
emotion_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", return_all_scores=True)

# Load the Excel file
df = pd.read_excel('/home/sambeg/US_tweets_predicted/US_tweets_final_predicted.xlsx')

# Step 1: Predict emotions using the RoBERTa model
def predict_emotion(text):
    try:
        # Get predictions from the RoBERTa pipeline
        predictions = emotion_pipeline(text)
        # Extract the label with the highest score
        max_label = max(predictions, key=lambda x: x['score'])['label']
        return max_label
    except Exception as e:
        return "unknown"  # Handle any errors during prediction

# Step 2: Apply the emotion prediction function
df['predicted_emotion'] = df['raw_text'].apply(predict_emotion)

# Step 3: Filter top 3 states with the most rows
top_states = df['state'].value_counts().head(3).index.tolist()

# Step 4: Plot emotion trends for each top state
for state in top_states:
    state_data = df[df['state'] == state]
    
    # Group data by Year_Month and predicted_emotion
    emotion_trends = state_data.groupby(['Year_Month', 'predicted_emotion']).size().reset_index(name='count')

    # Pivot the data to show emotions per month
    pivot_data = emotion_trends.pivot(index='Year_Month', columns='predicted_emotion', values='count').fillna(0)

    # Normalize the counts to proportions for each month
    pivot_data_percentage = pivot_data.div(pivot_data.sum(axis=1), axis=0)

    # Plot the trends as proportions
    plt.figure(figsize=(12, 8))
    for emotion in pivot_data_percentage.columns:
        plt.plot(pivot_data_percentage.index.astype(str), pivot_data_percentage[emotion], marker='o', label=emotion)

    # Add chart details
    plt.title(f"Emotion Trends Over Time (Proportion) in {state}")
    plt.xlabel("Year and Month")
    plt.ylabel("Proportion of Emotion Labels (0 to 1)")
    plt.xticks(rotation=45)
    plt.legend(title="Predicted Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_file = f"/home/sambeg/results_confusion_matrix_line_chart/{state}_proportion_emotion_trend.png"
    plt.savefig(output_file)
    plt.close()
