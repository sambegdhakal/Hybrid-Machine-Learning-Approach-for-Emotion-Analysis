import pandas as pd
import emoji
import re
import wordninja
import nltk
import openpyxl
import spacy
import contractions
import time

# Load the English model
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords

tweet_count = 0




# emoticon to emotion
emoticon_to_emotion = {
    ':)': 'happy',
    ':(': 'sad',
    ':D': 'laughing',
    ':|': 'neutral',
    ':/': 'confused',
    ':O': 'surprised',
    ':*': 'kiss',
    ':\')': 'tear of joy',
    '<3': 'love'
}


# Define the preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        # Convert non-string types to string
        text = str(text)

    global tweet_count
    # Increment the tweet count
    tweet_count += 1
    # Print the total number of lines processed
    print("Tweet number:", tweet_count)
    
    #convert emoticons to emotion texts
    for emoticon, emotion in emoticon_to_emotion.items():
        text = text.replace(emoticon, emotion)
        
    #convert emojis to emotion texts
    
    text= emoji.demojize(text)
    text= text.replace(' face:', '. ')
    text= text.replace(' :', '. ')
    text= text.replace('_', ' ')
    
    #expand contractions like won't to will not, couldn't to could not
    text = contractions.fix(text)

    # Remove links (URLs)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove consecutive periods (more than one)
    text = re.sub(r'…', ' ', text)

    # Remove multiple hyphens
    text = re.sub(r'--+', ' ', text)

    # Define a regular expression pattern to match allowed characters
    pattern = r'[^A-Za-z0-9\s.,!?#-]'

    # Remove characters that do not match the pattern
    cleaned_sentence = re.sub(pattern, ' ', text)

    # Remove new line characters
    cleaned_sentence= cleaned_sentence.replace("\r", "").replace("\n", " ").replace("\n", " ")
    
    #convert to lower case
    cleaned_sentence= cleaned_sentence.lower()
    
    # Split the string into words
    words = cleaned_sentence.split()

    # Iterate through the words and apply wordninja.split() only to words after the '#'
    for i, word in enumerate(words):
        if word.startswith('#'):
            words[i] = '#' + ' '.join(wordninja.split(word[1:]))

    # Join the words into a sentence
    final_paragraph = ' '.join(words)

    # Replace hashtags with "fullstop"
    final_paragraph = re.sub(r'#', '. ', final_paragraph)
    doc = nlp(final_paragraph)
    base_form_words = []

    for token in doc:
        # Check if the token is a superlative or comparative adjective
        if token.tag_ in ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS','RB', 'RBR', 'RBS']:
            base_form_words.append(token.lemma_)
        else:
            base_form_words.append(token.text)
    
    corrected_sentence=' '.join(base_form_words)

    # Remove extra spaces
    #corrected_sentence = " ".join(corrected_sentence.split())
    corrected_sentence = corrected_sentence.replace(", .", ".")
    corrected_sentence = corrected_sentence.replace(" .", ".")

    # Ensure the sentence ends with a period
    if not corrected_sentence.endswith("."):
        corrected_sentence += "."

    # Remove a leading period if it exists
    if corrected_sentence.startswith("."):
        corrected_sentence = corrected_sentence[1:]

    # Remove leading spaces, if any
    if corrected_sentence.startswith(" "):
        corrected_sentence = corrected_sentence.lstrip()

    corrected_sentence = corrected_sentence.replace(" ,", ",")
    
    corrected_sentence=re.sub(r'C ovid 19', ' Covid19', corrected_sentence)
    corrected_sentence = re.sub(r'C OVID 19', ' COVID19', corrected_sentence)
    corrected_sentence = re.sub(r'c ovid 19', ' covid19', corrected_sentence)
    corrected_sentence = re.sub(r'C ovid', ' Covid', corrected_sentence)
    corrected_sentence = re.sub(r'C OVID', ' COVID', corrected_sentence)
    corrected_sentence = re.sub(r'c ovid', ' covid', corrected_sentence)
    corrected_sentence = re.sub(r'W fh', ' WFH ', corrected_sentence)
    corrected_sentence = re.sub(r'W FH', ' WFH ', corrected_sentence)
    corrected_sentence = re.sub(r'w FH', ' WFH ', corrected_sentence)
    corrected_sentence = re.sub(r'w fh', ' WFH ', corrected_sentence)
    corrected_sentence = re.sub(r'wfh', ' WFH ', corrected_sentence)
    

    # Remove specific symbols
    symbols_to_remove = ['&amp;', '$', '%' , '& amp']
    for symbol in symbols_to_remove:
        corrected_sentence = corrected_sentence.replace(symbol, "")

    # Remove multiple consecutive periods at the end of the sentence
    corrected_sentence = re.sub(r'\.\.+', '.', corrected_sentence)

    # Remove whitespace before period
    corrected_sentence = re.sub(r' \.', '.', corrected_sentence)

    # Remove multiple spaces after a period
    corrected_sentence = re.sub(r'\.\s+', '. ', corrected_sentence)

    corrected_sentence = corrected_sentence.replace("?.", "?")
    corrected_sentence = corrected_sentence.replace("!.", "!")

    corrected_sentence = re.sub(r'! +', '!', corrected_sentence)
    corrected_sentence = re.sub(r'!+', '!', corrected_sentence)
    corrected_sentence = re.sub(r'\s+!', '!', corrected_sentence)
    corrected_sentence = re.sub(r'!', '! ', corrected_sentence)
    
    #Replace amp with and
    corrected_sentence= corrected_sentence.replace(" amp ", " and ")

    # add one whitespace before and after fullstop
    corrected_sentence= corrected_sentence.replace(".", " . ")
    corrected_sentence= " " + corrected_sentence

    # add one whitespace before and after exclamation mark
    corrected_sentence= corrected_sentence.replace("!", " ! ")
    corrected_sentence= " " + corrected_sentence

    # one whitespace before and after comma
    corrected_sentence= corrected_sentence.replace(",", " , ")
    corrected_sentence= " " + corrected_sentence

    # one whitespace before and after comma
    corrected_sentence= corrected_sentence.replace("?", " ? ")
    corrected_sentence= " " + corrected_sentence

    return corrected_sentence



# Path to your Excel file
excel_file_path = '/convert_to_Excel/us-tweets.xlsx'    


# Read the Excel file
data = pd.read_excel(excel_file_path)


# column to preproces
column_name = 'raw_text'

#convert datetime to date
#data['created_at'] = pd.to_datetime(data['created_at']).dt.date


# Capture start time
start_time = time.time()

# Apply the preprocessing function to the specified column
data[column_name] = data[column_name].apply(preprocess_text)

# Capture end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

print(f"Preprocessing started at: {start_time}")
print(f"Preprocessing ended at: {end_time}")
print(f"Total time taken for preprocessing: {duration} seconds")

# Specify the Excel file name (e.g., 'preprocessed_data.xlsx')
output_excel_file = '/preprocessed-us-tweets/us_tweets_preprocess.xlsx'

# Save the preprocessed DataFrame to an Excel file
data.to_excel(output_excel_file, index=False, engine='openpyxl')
