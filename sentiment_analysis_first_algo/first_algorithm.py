import pandas as pd
import os
import csv
import re
import nltk
nltk.download('punkt')  # Download the necessary data for tokenization
from nltk.tokenize import sent_tokenize

#Load data
excel_file = "/home/sambeg/preprocessed-us-tweets/us_tweets_preprocess.xlsx"
positive_emotions= ['happy', 'calm']
negative_emotions= ['sad', 'angry', 'annoyed', 'nervous', 'fear']

#read excel file
df = pd.read_excel(excel_file)
#sentences = df["raw_text"].tolist()



#read emotions from csv files
def read_emotion_phrases_from_csv(filename):
    phrases = []
    with open(filename, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            phrases.extend([phrase.lower() for phrase in row])
    return phrases

 # Check if negation found
#def is_negation(sentence, negation_words):
    
#   return any(phrase.lower() in sentence.lower() for phrase in negation_words)



#emotional scores calculation
def detect_emotion(text, emotion_directory):
    emotion_files = {
        "happy": os.path.join(emotion_directory, "happy.csv"),
        "sad": os.path.join(emotion_directory, "sad.csv"),
        "angry": os.path.join(emotion_directory, "angry.csv"),
        "annoyed": os.path.join(emotion_directory, "annoyed.csv"),
        "fear": os.path.join(emotion_directory, "fear.csv"),
        "nervous": os.path.join(emotion_directory, "nervous.csv"),
        "surprised": os.path.join(emotion_directory, "surprised.csv"),
        "calm": os.path.join(emotion_directory, "calm.csv"),
        "neutral": os.path.join(emotion_directory, "neutral.csv")
    }

    emotion_weights=1
    emotion_phrases = {}
    for emotion, filename in emotion_files.items():
        emotion_phrases[emotion] = read_emotion_phrases_from_csv(filename)

    

    #context keywords 
    context_keywords = {"but", "however", "although", "and"}

    #negation words
    negation_words = {" not ", " never ", " no ", " don't ", " doesn't ", " quite the opposite ",
                      " far from ", " quite distant from "}

    #replacing context keywords with "."    
    for keyword in context_keywords:
        text=text.replace(keyword,'.')

    #replace "everything but" and "anything but" with "not" 
    text = text.replace("everything but", "not")
    text = text.replace("anything but", "not")

    #replace phrase that contains negation which will make the word positive after phase
    #List of phrase
    list_of_phrase = [" can not prevent us from " , " will not stop us from " , " could not hold us back from " , " never will keep us from " , 
                      " will never prevent us from " , " will not hinder us from " , " can not stop us from " , " will not bar us from " , 
                      " will not stand in our way of " , " will not block our path to " , " will not keep us from " , " will not stand in our way ",
                      " can not impede our progress " , " will not delay us from " , " can not hold us back " , " will not thwart our effort " , 
                      " will not limit our ability " , " can not stifle our ambition " , " will not obstruct our path " , 
                      " will not derail our plan " , " can not diminish our resolve " , " will not interfere with our goal " , 
                      " can not suppress our drive " , " will not curtail our journey " , " will not undermine our effort " , 
                      " can not block our success " , " can not prevent me from " , " will not stop me from " , " could not hold me back from " , 
                      " never will keep me from " , " will never prevent me from " , " will not hinder me from " , " can not stop me from " , 
                      " will not bar me from " , " will not stand in my way of " , " will not block my path to " , " will not keep me from " , 
                      " will not stand in my way " , " can not impede my progress " , " will not delay me from " , " can not hold me back " , 
                      " will not thwart my effort " , " will not limit my ability " , " can not stifle my ambition " , 
                      " will not obstruct my path " , " will not derail my plan " , " can not diminish my resolve " , 
                      " will not interfere with my goal " , " can not suppress my drive " , " will not curtail my journey " , 
                      " will not undermine my effort " , " can not block my success " , " can not prevent you from " , 
                      " will not stop you from " , " could not hold you back from " , " never will keep you from " , 
                      " will never prevent you from " , " will not hinder you from " , " can not stop you from " , " will not bar you from " , 
                      " will not stand in your way of " , " will not block your path to " , " will not keep you from " , 
                      " will not stand in your way " , " can not impede your progress " , " will not delay you from " , " can not hold you back " ,
                      " will not thwart your effort " , " will not limit your ability " , " can not stifle your ambition " , 
                      " will not obstruct your path " , " will not derail your plan " , " can not diminish your resolve " , 
                      " will not interfere with your goal " , " can not suppress your drive " , " will not curtail your journey " , 
                      " will not undermine your effort " , " can not block your success " , " can not prevent them from " , 
                      " will not stop them from " , " could not hold them back from " , " never will keep them from " , 
                      " will never prevent them from " , " will not hinder them from " , " can not stop them from " , 
                      " will not bar them from " , " will not stand in their way of " , " will not block their path to " , 
                      " will not keep them from " , " will not stand in their way " , " can not impede their progress " , 
                      " will not delay them from " , " can not hold them back " , " will not thwart their effort " , 
                      " will not limit their ability " , " can not stifle their ambition " , " will not obstruct their path " , 
                      " will not derail their plan " , " can not diminish their resolve " , " will not interfere with their goal " , 
                      " can not suppress their drive " , " will not curtail their journey " , " will not undermine their effort " , 
                      " can not block their success " , " can not prevent him from " , " will not stop him from " , 
                      " could not hold him back from " , " never will keep him from " , " will never prevent him from " , 
                      " will not hinder him from " , " can not stop him from " , " will not bar him from " , " will not stand in his way of " , 
                      " will not block his path to " , " will not keep him from " , " will not stand in his way " , 
                      " can not impede his progress " , " will not delay him from " , " can not hold him back " , " will not thwart his effort " , 
                      " will not limit his ability " , " can not stifle his ambition " , " will not obstruct his path " , 
                      " will not derail his plan " , " can not diminish his resolve " , " will not interfere with his goal " , 
                      " can not suppress his drive " , " will not curtail his journey " , " will not undermine his effort " , 
                      " can not block his success " , " can not prevent her from " , " will not stop her from " , " could not hold her back from " , 
                      " never will keep her from " , " will never prevent her from " , " will not hinder her from " , " can not stop her from " , 
                      " will not bar her from " , " will not stand in her way of " , " will not block her path to " , " will not keep her from " , 
                      " will not stand in her way " , " can not impede her progress " , " will not delay her from " , " can not hold her back " , 
                      " will not thwart her effort " , " will not limit her ability " , " can not stifle her ambition " , 
                      " will not obstruct her path " , " will not derail her plan " , " can not diminish her resolve " , 
                      " will not interfere with her goal " , " can not suppress her drive " , " will not curtail her journey " , 
                      " will not undermine her effort " , " can not block her success " ]
    
    # Iterate over each phrase and replace "not" with an empty string when found in the sentence
    for phrase in list_of_phrase:
        if phrase in text:
            text = text.replace(phrase, phrase.replace("not ", ""))
    
    #Tokenize the sentences; separate into different sentences
    sentences= sent_tokenize(text)
    
    emotion_scores = {emotion: 0 for emotion in emotion_files}

    for sentence in sentences:
        sentence= " "+ sentence
        if sentence == "":
            continue

        else:  
            negation_found = any(phrase in sentence.lower() for phrase in negation_words)
            question_mark_found = sentence.strip().endswith('?')
            if negation_found:
                # Find the starting index and length of the first found negation phrase
                index, length = -1, 0
                for phrase in negation_words:
                    if phrase in sentence.lower():
                        index = sentence.lower().find(phrase)
                        length = len(phrase)
                    break  

                if index != -1:
                    #--not needed for now Extract the sentence after the negation phrase
                    # sentence_after_negation = sentence[index + length:].strip()
                    emotion_scores["neutral"] =1 
            
            elif question_mark_found:
                for emotion, phrases in emotion_phrases.items():
                    for phrase in phrases:
                        if phrase in sentence.lower():
                            emotion_scores[emotion] += 0.5
                
            else:
                for emotion, phrases in emotion_phrases.items():
                    
                    for phrase in phrases:
                        if phrase in sentence.lower():
                            #print(phrase)
                            emotion_scores[emotion] += emotion_weights
                        
        #calculate total emotion score
        total_emotion_score = sum(emotion_scores.values()) 
        
    
    
    if total_emotion_score == 0:
        total_emotion_score=1
        emotion_scores["neutral"]=1
      
    
    #Calculate the score for each emotion as a ratio of its score to the total emotion score
    for emotion in emotion_scores:
        if emotion!="neutral": 
            if emotion_scores[emotion]!=0:
                emotion_scores["neutral"]=0;
        emotion_scores[emotion] = emotion_scores[emotion] / total_emotion_score
        
        # Find the maximum absolute value
        max_abs_value = max(abs(score) for score in emotion_scores.values())

        # Find all emotions with the maximum absolute value
        emotions = [emotion for emotion, score in emotion_scores.items() if abs(score) == max_abs_value]

        # Check if 'happy' and 'angry' or 'happy' and 'sad' are in the list and replace them with 'neutral'
        if ('happy' in emotions and ('angry' in emotions or 'sad' in emotions or 'annoyed' in emotions or 'fear' in emotions or 'surprised' in emotions or 'nervous' in emotions)):
            emotions = ['neutral']
        elif ('calm' in emotions and ('angry' in emotions or 'sad' in emotions or 'annoyed' in emotions or 'fear' in emotions or 'surprised' in emotions or 'nervous' in emotions)):
            emotions = ['neutral']  
        # to only have one emotions, giving an emotion with higher negative value a priority
        elif ('happy' in emotions and ('calm' in emotions or 'surprised' in emotions)):
            emotions = ['happy']

        elif ('calm' in emotions and 'surprised' in emotions):
            emotions = ['calm']

        elif ('angry' in emotions and ('sad' in emotions or 'annoyed' in emotions or 'fear' in emotions or 'surprised' in emotions or 'nervous' in emotions)):    
            emotions = ['angry']

        elif ('annoyed' in emotions and ('sad' in emotions or 'fear' in emotions or 'surprised' in emotions or 'nervous' in emotions)):
            emotions = ['annoyed']

        elif ('fear' in emotions and ('sad' in emotions or 'surprised' in emotions or 'nervous' in emotions)):
            emotions = ['fear']

        elif ('sad' in emotions and ('surprised' in emotions or 'nervous' in emotions)):
            emotions = ['sad']
        
        elif ('nervous' in emotions and 'surprised' in emotions):
            emotions = ['nervous']

    return emotions              
                   


# Example usage:
dataset_directory = '/home/sambeg/Datasets/'

for index, row in df.iterrows():
    emotions= detect_emotion(row["raw_text"], dataset_directory) 
    detected_emotions=emotions
    #labelling text as positive, negative or neutral
    for emotion in emotions:
        if (emotion in positive_emotions):
            label = 'Positive'
            break;
        elif (emotion in negative_emotions):
            label = 'Negative'
            break;
        else:
            label = 'Neutral'
            break;
    df.at[index, 'labels_algo'] = label
    
    # Add the detected emotions to a new column in the DataFrame
    df.at[index , 'detected_emotion'] = detected_emotions




# Save the DataFrame to a new Excel file
output_excel_file = "/home/sambeg/Labelled_using_algo/labelled_first_algo.xlsx"
df.to_excel(output_excel_file, index=False)

    



