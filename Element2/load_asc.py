import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to assign sample type
def assign_sample_type(sample_id):
    if "std" in sample_id.lower():  # Convert to lowercase for case-insensitive check
        return "standard"
    if "bl" in sample_id.lower():  # Convert to lowercase for case-insensitive check
        return "blank"
    else:
        return "sample"
            
def load_asc(file_name):
    """ This function convert a Thermo Element2 .ASC file and return it as a 
    pandas dataframe.

    Params
    ------
    file_name : string
        The file name of .asc file example: yda080909.ASC
    
    Returns
    -------
    df_merge : pandas dataframe
        Pandas dataframe containing sample_id, type_, intensity, rsd
    """
    # Load asc data and skip empty rows

    row_num, col_num = pd.read_csv(file_name, delimiter='\t').shape

    skip_rows = [1, 2, 3, 4, 5, row_num-3, row_num-2, row_num-1, row_num,
                row_num+1, row_num+2]
    column = [x for x in range(0, col_num-1) if x % 2 == 1 or x == 1 or x == 0]
    df = pd.read_csv(file_name, delimiter='\t', usecols=column, skiprows=skip_rows)

    # separate signal and rsd
    sample_name = [col for i, col in enumerate(df.columns) if i%2==1 or i==0]
    rsd = [col for i, col in enumerate(df.columns) if i%2==0 or i==0]
    # signal
    df_signal = df[sample_name]
    
    # rsd, copy analyte name to rsd, append _rsd
    df_rsd = df[rsd]
    df_rsd = df_rsd.rename(columns=dict(zip(df_rsd.columns, sample_name)))
    df_rsd['Unnamed: 0'] = df_rsd['Unnamed: 0'].apply(lambda x: x + "_rsd")

    # Transpose dataframe
    df_rsd_T = df_rsd.set_index('Unnamed: 0').T
    df_signal_T = df_signal.set_index('Unnamed: 0').T

    # Merge signal and rsd
    df_merge = df_signal_T.join(df_rsd_T, how='inner')
    df_merge.insert(0, 'Sample_ID', df_merge.index)
    df_merge = df_merge.reset_index(drop=True)

    # Function to assign sample type
    def assign_sample_type(sample_id):
        if "std" in sample_id.lower():  # Convert to lowercase for case-insensitive check
            return "standard"
        if "bl" in sample_id.lower():  # Convert to lowercase for case-insensitive check
            return "blank"
        else:
            return "sample"

    df_merge['Sample_Type'] = df_merge['Sample_ID'].apply(assign_sample_type)


    # Function to assign standard id
    def assign_standard_type(sample_id):
        if "std_1x" in sample_id.lower(): 
            return "std_1x"
        if "std_2" in sample_id.lower(): 
            return "std_2"
        if "std_3" in sample_id.lower(): 
            return "std_3"
        if "std_4" in sample_id.lower(): 
            return "std_4"
        if "std_5" in sample_id.lower(): 
            return "std_5"
        if "std_6" in sample_id.lower(): 
            return "std_6"
        if "std_1" in sample_id.lower(): 
            return "std_1"
        else:
            return "non_std"

    df_merge['Standard_Type'] = df_merge['Sample_ID'].apply(assign_standard_type)

    return df_merge

    

if __name__ == "__main__":
# Initialize Tkinter and hide the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select the .ASC file
    filename_asc = filedialog.askopenfilename(title="Select the ASC file", filetypes=[("ASC files", "*.ASC"), ("All files", "*.*")])
    
    if filename_asc:
        # Prompt the user for the output CSV file name
        file_csv = input("The filename for the converted file (without extension): ")
        file_csv_ext = file_csv + ".csv"

        # Process the .ASC file and save the output as a CSV file
        df = load_asc(filename_asc)
        df.to_csv(file_csv_ext, index=None)  # Save the DataFrame to CSV without the index
        print(f"File saved as {file_csv_ext}")
    else:
        print("No file selected.")