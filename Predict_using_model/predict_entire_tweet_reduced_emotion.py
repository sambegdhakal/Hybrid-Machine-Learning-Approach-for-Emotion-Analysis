import pandas as pd
import geopandas as gpd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from shapely.geometry import Point

# trained model batch4 loaded
loaded_model = tf.keras.models.load_model('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned.h5')

# sample loaded
df = pd.read_excel('/home/sambeg/reduced_emotions/preprocessed-us-tweets/us_tweets_preprocess.xlsx')
sample_data = df['raw_text']

# word_index loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_wordindex.pkl', 'rb') as file:
    word_index = pickle.load(file)

# max_sequence_len loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_max_sequence_len.pkl', 'rb') as file:
    max_sequence_len = pickle.load(file)

# emotion_label_encoder loaded
with open('/home/sambeg/reduced_emotions/Neural_network_trained_models/Hyperparameter_tuned_models/Emotionbhav_hptuned_emotion_label_encoder.pkl', 'rb') as file:
    emotion_label_encoder = pickle.load(file)

# Tokenization
def tokenize(sentences, word_index):
    sequences = []
    for sentence in sentences:
        sequence = []
        for word in sentence.split():
            if word in word_index:
                sequence.append(word_index[word])
        sequences.append(sequence)
    return sequences


sequences = tokenize(sample_data, word_index)
padded_sequences = np.array([np.pad(seq, (0, max_sequence_len - len(seq))) for seq in sequences])


predictions = loaded_model.predict(padded_sequences)

# Convert predictions to labels
predicted_labels = np.argmax(predictions, axis=1)
predicted_emotions = emotion_label_encoder.inverse_transform(predicted_labels)

predicted_cleaned = []
# Add predictions to dataframe and save
for predicted in predicted_emotions:
    predicted_cleaned.append(predicted.replace("['", "").replace("']", ""))

total_size= df['raw_text'].size


#predicted and cleaned
df['predicted_emotion'] = predicted_cleaned

# Load the shapefile (replace with your shapefile path)
states = gpd.read_file("cb_2018_us_state_500k.shp")

# Assuming your DataFrame 'df' contains 'lat' and 'long' columns
# Create Point geometries for latitude and longitude
df['geometry'] = df.apply(lambda row: Point(row['long'], row['lat']), axis=1)

# Convert the DataFrame into a GeoDataFrame
geo_df = gpd.GeoDataFrame(df, geometry='geometry')

# Ensure both GeoDataFrames use the same CRS (Coordinate Reference System)
geo_df = geo_df.set_crs(states.crs, allow_override=True)

# Perform spatial join to find which state each point belongs to
merged = gpd.sjoin(geo_df, states, how="left", predicate="intersects")

# Extract the state name (replace 'NAME' with the appropriate column name in your shapefile)
df['state'] = merged['NAME']

# remove time 
df['Year_Month'] = pd.to_datetime(df['created_at']).dt.to_period('M')

#list needed columns
needed_columns = ['raw_text', 'Year_Month', 'state', 'predicted_emotion']

# save in predicted file
df[needed_columns].to_excel('/home/sambeg/reduced_emotions/US_tweets_predicted/US_tweets_final_predicted.xlsx', index=False)





