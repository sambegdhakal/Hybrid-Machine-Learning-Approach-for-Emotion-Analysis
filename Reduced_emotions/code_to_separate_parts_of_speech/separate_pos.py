import csv
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def classify_words(words):
    nouns, verbs, adjectives, adverbs , notclassified = [], [], [], [], []

    words = [word.strip() for word in words if word.strip()] 

    for word in words:
        doc = nlp(word)
        classified = False
        for token in doc:
            if token.pos_ == "NOUN":
                nouns.append(word)
                classified = True
            elif token.pos_ == "VERB":
                verbs.append(word)
                classified = True
            elif token.pos_ == "ADJ":
                adjectives.append(word)
                classified = True
            elif token.pos_ == "ADV":
                adverbs.append(word)
                classified = True
                
        if not classified:
            notclassified.append(word)    
    return nouns, verbs, adjectives, adverbs, notclassified

emotions = ['happy', 'sad', 'angry', 'annoyed', 'fear', 'neutral', 'surprised']
part_of_speech = ['nouns', 'verbs', 'adjectives', 'adverbs', 'notclassified']
base_path = "/home/sambeg/reduced_emotions/Datasets/"

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
        nouns = list(set(nouns))
        verbs = list(set(verbs))
        adjectives = list(set(adjectives))
        adverbs = list(set(adverbs))
        notclassified = list(set(notclassified))

        # Add space before and after each classified word
        nouns = [f" {w} " for w in nouns]
        verbs = [f" {w} " for w in verbs]
        adjectives = [f" {w} " for w in adjectives]
        adverbs = [f" {w} " for w in adverbs]
        notclassified = [f" {w} " for w in notclassified]

    for pos in part_of_speech:
        output_path="/home/sambeg/reduced_emotions/separated_pos_datasets/"
        output_file = f"{output_path}{pos}_{emotion}.csv"
        # Write classified words to separate files
        with open(output_file, 'w') as file:
            writer = csv.writer(file)
            if(pos == "nouns"):
                writer.writerow(nouns)
            elif(pos == "verbs"):
                writer.writerow(verbs)
            elif(pos == "adjectives"):
                writer.writerow(adjectives)
            elif(pos == "adverbs"):
                writer.writerow(adverbs)
            elif(pos == "notclassified"):
                writer.writerow(notclassified)                
        print(f"File {output_file} has been written")

print("Words have been classified and written to separate files.")
