def remove_duplicates(input_file, output_file): 
    distinct_values = set()

    with open(input_file, 'r') as infile:
        text = infile.read()
        words = {word.strip() for word in text.split(',') if word.strip()}

    distinct_values.update(words)

    with open(output_file, 'w') as outfile:
        data_to_write = ' , '.join(distinct_values)
        data_to_write = ' ' + data_to_write + ' '
        outfile.write(data_to_write)

    return distinct_values

def count_words(input_file):
    with open(input_file, 'r') as infile:
        text = infile.read()
        words = [word.strip() for word in text.split(',') if word.strip()]
    return len(words)

emotions = ['happy', 'sad', 'angry', 'annoyed', 'fear', 'neutral', 'surprised']
total_word_count = 0
global_distinct_words = set()

for emotion in emotions:
    input_file = f"{emotion}.csv"
    output_file = f"{emotion}.csv"
    
    distinct_words = remove_duplicates(input_file, output_file)
    global_distinct_words.update(distinct_words)
    
    print(f"Distinct values from {input_file} have been written to {output_file}.")
    
    with open(output_file, 'r') as file:
        content = file.read()

    if content.startswith("  ,"):
        content = content.lstrip("  ,")
    
    content = content.replace(",   ,", ",")
    
    with open(output_file, 'w') as file:
        file.write(content)
    
    word_count = count_words(output_file)
    total_word_count += word_count
    print(f"Word count in {output_file}: {word_count}")

print(f"Total word count across all files: {total_word_count}")
print(f"Total distinct words across all files: {len(global_distinct_words)}")
