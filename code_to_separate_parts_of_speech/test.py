import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def classify_words(words):
    nouns, verbs, adjectives, adverbs, notclassified = [], [], [], [], []
    
    # Join words into a sentence for better POS tagging
    sentence = " ".join(words)
    doc = nlp(sentence)  # Process the sentence with spaCy
    
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.append(token.text)
        elif token.pos_ == "VERB":
            verbs.append(token.text)
        elif token.pos_ == "ADJ":
            adjectives.append(token.text)
        elif token.pos_ == "ADV":
            adverbs.append(token.text)
        else:
            notclassified.append(token.text)
    
    return nouns, verbs, adjectives, adverbs, notclassified

# Test words
words_to_classify = ['mournful', 'throwback', 'run', 'quickly', 'happiness']

# Call the classify_words function
nouns, verbs, adjectives, adverbs, notclassified = classify_words(words_to_classify)

# Output the results
print("Nouns:", nouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)
print("Adverbs:", adverbs)
print("Not Classified:", notclassified)
