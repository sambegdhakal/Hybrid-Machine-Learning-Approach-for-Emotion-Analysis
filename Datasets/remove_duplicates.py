def remove_duplicates(input_file, output_file):
    distinct_values = set()

    with open(input_file, 'r') as infile:
        text = infile.read()
        words = [word.strip() for word in text.split(',')]

    # adding distinct words in set
    for word in words:
        distinct_values.add(word)

    # values wrriten in each output file as a comma separated values
    with open(output_file, 'w') as outfile:
        data_to_write= ' , '.join(distinct_values)
        data_to_write= ' ' + data_to_write + ' '
        outfile.write(data_to_write)

#set of 9 emotions for full emotions datasets
emotions = ['happy', 'calm', 'sad', 'angry', 'annoyed', 'nervous', 'fear', 'neutral', 'surprised']

for emotion in emotions:
    if __name__ == "__main__":
        input_file = f"{emotion}.csv"
        output_file = f"{emotion}.csv"
        remove_duplicates(input_file, output_file)
        print(f"Distinct values from {input_file} have been written to {output_file}.")

    with open(output_file, 'r') as file:
         content = file.read()

    if content.startswith("  ,"):
        content = content.lstrip("  ,")   

    content = content.replace(",   ,", ",")

    with open(output_file, 'w') as file:
        file.write(content)          
