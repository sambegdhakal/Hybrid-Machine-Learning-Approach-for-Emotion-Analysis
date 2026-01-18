import requests

def get_synonyms(word):
    response = requests.get(f'https://api.datamuse.com/words?rel_syn={word}')
    if response.status_code == 200:
        words = response.json()
        synonyms = [entry['word'] for entry in words]
        return synonyms
    else:
        return f"Error: {response.status_code}"

# use this code
word = 'annoying'
synonyms = get_synonyms(word)
print(f"Synonyms for '{word}': {', '.join(synonyms)}")
