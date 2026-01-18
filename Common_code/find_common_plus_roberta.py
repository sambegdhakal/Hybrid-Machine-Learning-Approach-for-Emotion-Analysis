import pandas as pd
df = pd.read_excel('/home/sambeg/labelling/labelled_second_roberta.xlsx')

matching_rows = df[(df['detected_emotion']==df['detected_emotion_2nd']) & (df['labels_algo'].str.lower()==df['sentiment_roberta'].str.lower()) ]


matching_rows.drop(['labels_algo2', 'detected_emotion_2nd'], axis=1)

matching_rows.to_excel('/home/sambeg/labelling/matched_first_second_roberta.xlsx')

