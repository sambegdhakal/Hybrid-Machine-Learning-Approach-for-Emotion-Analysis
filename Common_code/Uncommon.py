import pandas as pd
df = pd.read_excel('/home/sambeg/algo_labelling/labelled_second.xlsx')

# Remove unwanted characters "['" and "']" from both columns
df['detected_emotion'] = df['detected_emotion'].str.replace(r"['\[\]]", '', regex=True)
df['detected_emotion_2nd'] = df['detected_emotion_2nd'].str.replace(r"['\[\]]", '', regex=True)

unmatched_rows = df[df['detected_emotion']!=df['detected_emotion_2nd']]

unmatched_rows.drop(['labels_algo', 'detected_emotion' , 'labels_algo2', 'detected_emotion_2nd'], axis=1)

unmatched_rows.to_excel('/home/sambeg/algo_labelling/unmatched_first_second.xlsx')



