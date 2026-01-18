import pandas as pd

df = pd.read_excel("/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx")

positive_emotions= ['happy']
negative_emotions= ['sad', 'angry', 'annoyed', 'fear']


def emotion_condition(predicted_emotion):
    if (predicted_emotion.lower() in positive_emotions):
        return "Positive"
    elif (predicted_emotion.lower() in negative_emotions):
        return "Negative"
    else:
        return "Neutral"
    
df['predicted_sentiment'] = df['predicted_emotion'].apply(emotion_condition)

output_excel_file = "/home/sambeg/reduced_emotions/human_labelled/human_labeled_predicted.xlsx"
df.to_excel(output_excel_file, index=False)