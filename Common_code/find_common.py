import pandas as pd
df = pd.read_excel('/home/sambeg/Labelled_using_algo/labelled_first_second.xlsx')

# Remove unwanted characters "['" and "']" from both columns
df['detected_emotion'] = df['detected_emotion'].str.replace(r"['\[\]]", '', regex=True)
df['detected_emotion_2nd'] = df['detected_emotion_2nd'].str.replace(r"['\[\]]", '', regex=True)


matching_rows = df[df['detected_emotion']==df['detected_emotion_2nd']]


matching_rows.drop(['labels_algo2', 'detected_emotion_2nd'], axis=1)

matching_rows.to_excel('/home/sambeg/Labelled_using_algo/matched_first_second.xlsx')

