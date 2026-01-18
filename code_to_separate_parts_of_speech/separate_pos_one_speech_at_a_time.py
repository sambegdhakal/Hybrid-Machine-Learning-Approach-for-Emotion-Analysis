import csv
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def classify_words(words):
    # Priority order: adjective > adverb > verb > noun > not classified
    adjectives, adverbs, verbs, nouns, notclassified = [], [], [], [], []
    
    for word in words:
        doc = nlp(word)
        classified = False
        for token in doc:
            if token.pos_ == "ADJ" and not classified:
                adjectives.append(word)
                classified = True
            elif token.pos_ == "ADV" and not classified:
                adverbs.append(word)
                classified = True
            elif token.pos_ == "VERB" and not classified:
                verbs.append(word)
                classified = True
            elif token.pos_ == "NOUN" and not classified:
                nouns.append(word)
                classified = True
        if not classified:
            notclassified.append(word)
    
    return nouns, verbs, adjectives, adverbs, notclassified

# Emotions and part of speech
emotions = ['happy', 'calm', 'sad', 'angry', 'annoyed', 'nervous', 'fear', 'neutral', 'surprised']
part_of_speech = ['nouns', 'verbs', 'adjectives', 'adverbs', 'notclassified']
base_path = "/home/sambeg/Datasets/"

for emotion in emotions:
    # Read words from CSV
    input_file = f"{base_path}{emotion}.csv"
    print(input_file)
    words = []
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            words.extend(row)

    # Classify words
    nouns, verbs, adjectives, adverbs, notclassified = classify_words(words)

    for pos in part_of_speech:
        output_path = "/home/sambeg/separated_pos_datasets_only_one_speech_at_a_time/"
        output_file = f"{output_path}{pos}_{emotion}.csv"
        # Write classified words to separate files
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            if pos == "nouns":
                writer.writerow(nouns)
            elif pos == "verbs":
                writer.writerow(verbs)
            elif pos == "adjectives":
                writer.writerow(adjectives)
            elif pos == "adverbs":
                writer.writerow(adverbs)
            elif pos == "notclassified":
                writer.writerow(notclassified)
        print(f"File {output_file} has been written")

print("Words have been classified and written to separate files.")
