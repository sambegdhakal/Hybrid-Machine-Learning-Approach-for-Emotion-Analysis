import pandas as pd
df = pd.read_excel('/home/sambeg/labelling/labelled_second_vader.xlsx')

unmatched_rows = df[(df['detected_emotion']!=df['detected_emotion_2nd']) | (df['labels_algo'].str.lower()!=df['sentiment_vader'].str.lower())]

unmatched_rows.drop(['labels_algo', 'detected_emotion' , 'labels_algo2', 'detected_emotion_2nd', 'sentiment_vader'], axis=1)

unmatched_rows.to_excel('/home/sambeg/labelling/unmatched_first_second_vader.xlsx')



