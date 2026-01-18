import pandas as pd
import matplotlib.pyplot as plt

# Load the excel file with predicted emotions
df = pd.read_excel('/home/sambeg/reduced_emotions/US_tweets_predicted/US_tweets_final_predicted.xlsx')

# Top 3 states with high row counts
top_states = df['state'].value_counts().head(3).index.tolist()

# Line plots for each top 3 states
for state in top_states:
    state_data = df[df['state'] == state]  # Filter state

    # Count rows by grouping year_month and predicted emotion
    emotion_trends = state_data.groupby(['Year_Month', 'predicted_emotion']).size().reset_index(name='count')

    # Pivot the data
    pivot_data = emotion_trends.pivot(index='Year_Month', columns='predicted_emotion', values='count').fillna(0)

    # Normalize the counts to percentages (0 to 1) for each month
    pivot_data_percentage = pivot_data.div(pivot_data.sum(axis=1), axis=0)

    # Plot the line chart for each emotion
    plt.figure(figsize=(12, 8))
    for emotion in pivot_data_percentage.columns:
        plt.plot(pivot_data_percentage.index.astype(str), pivot_data_percentage[emotion], marker='o', label=emotion)

    # Add chart details
    plt.title(f"Emotion Trends Over Time in {state} (Normalized to Percentage)")
    plt.xlabel("Year and Month")
    plt.ylabel("Emotion Percentage (0-1)")
    plt.xticks(rotation=45)
    plt.legend(title="Predicted Emotions", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_file = f"/home/sambeg/results_confusion_matrix_line_chart/{state}_reduced_emotion_trend_percentage.png"
    plt.savefig(output_file)
    plt.close()
