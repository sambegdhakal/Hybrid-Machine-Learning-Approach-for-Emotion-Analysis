import pandas as pd


input_file = "/home/sambeg/Labelled_using_algo/labelled_first_algo.xlsx"  
output_file = "/home/sambeg/Labelled_using_algo/Predicted_data_first_algo.xlsx"

df = pd.read_excel(input_file) #read input file

column1 = df['raw_text']
column2 = df['detected_emotion'].str.replace("[","").str.replace("]","").str.replace("'","")


columns=pd.DataFrame({
    'raw_text': column1,
    'detected_emotion': column2
})

columns.to_excel(output_file, index=False)