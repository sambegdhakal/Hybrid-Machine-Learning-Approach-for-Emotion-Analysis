def check_duplicates(file_path):
    words = []

    with open(file_path, 'r') as file:
        text = file.read()
        words = [word.strip() for word in text.split(',')]

    # keeping track of unique words using set
    seen = set()
    duplicates = set()

    for word in words:
        if word in seen:
            duplicates.add(word)
        else:
            seen.add(word)

    return duplicates
emotions = ['happy', 'calm', 'sad', 'angry', 'annoyed', 'nervous', 'fear', 'neutral', 'surprised']

for emotion in emotions:
    if __name__ == "__main__":
        file_path = f"{emotion}.csv"
        duplicates = check_duplicates(file_path)
        if duplicates:
            print("Duplicates found:")
            print(duplicates)
        else:
            print(f"No duplicates found for {file_path}")
