import pandas as pd

# Load the Excel files
file1 = '/home/sambeg/labelling/labelled_second.xlsx'  # Replace with your first file path
file2 = '/home/sambeg/convert_to_Excel/us-tweets.xlsx'  # Replace with your second file path

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

tweets = pd.merge(df2[['tweet_id', 'raw_text']],df1['tweet_id'] , on='tweet_id')

text= tweets['raw_text']

text.to_excel('/home/sambeg/labelling/unmatched_with_emojis.xlsx')