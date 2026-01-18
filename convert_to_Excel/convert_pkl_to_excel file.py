import pandas as pd
import os

current_dir = os.getcwd()
pkl_file_path = '/home/sambeg/raw-us-tweets/us-tweets.pkl'
output_file = os.path.join(current_dir, 'us-tweets.xlsx')

try:
    # Load the .pkl file
    df = pd.read_pickle(pkl_file_path)
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    
    print(f'Excel file saved at: {output_file}')
    
except FileNotFoundError:
    print(f'Error: The file {pkl_file_path} does not exist.')
except Exception as e:
    print(f'An error occurred: {e}')
