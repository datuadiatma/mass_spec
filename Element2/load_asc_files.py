import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def assign_sample_type(sample_id):
    if "std" in sample_id.lower():
        return "standard"
    if "bl" in sample_id.lower():
        return "blank"
    else:
        return "sample"

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

def load_asc_files(file_names):
    """
    Process multiple Thermo Element2 .ASC files by first merging all signals,
    then all RSDs, and finally combining them.
    
    Parameters
    ----------
    file_names : list
        List of .ASC file paths to process
    
    Returns
    -------
    pandas.DataFrame
        Combined dataframe containing data from all input files
    """
    all_signals = []
    all_rsds = []
    
    for file_name in file_names:
        try:
            # Get dimensions of the file
            row_num, col_num = pd.read_csv(file_name, delimiter='\t').shape
            
            # Define rows to skip and columns to use
            skip_rows = [1, 2, 3, 4, 5, row_num-3, row_num-2, row_num-1, row_num,
                        row_num+1, row_num+2]
            column = [x for x in range(0, col_num-1) if x % 2 == 1 or x == 1 or x == 0]
            
            # Read the file
            df = pd.read_csv(file_name, delimiter='\t', usecols=column, skiprows=skip_rows)
            
            # Separate signal and rsd columns
            sample_name = [col for i, col in enumerate(df.columns) if i%2==1 or i==0]
            rsd = [col for i, col in enumerate(df.columns) if i%2==0 or i==0]
            
            # Process signal data
            df_signal = df[sample_name]
            df_signal_T = df_signal.set_index('Unnamed: 0').T
            df_signal_T.insert(0, 'Sample_ID', df_signal_T.index)
            df_signal_T = df_signal_T.reset_index(drop=True)
            
            # Process RSD data
            df_rsd = df[rsd]
            df_rsd = df_rsd.rename(columns=dict(zip(df_rsd.columns, sample_name)))
            df_rsd['Unnamed: 0'] = df_rsd['Unnamed: 0'].apply(lambda x: x + "_rsd")
            df_rsd_T = df_rsd.set_index('Unnamed: 0').T
            df_rsd_T.insert(0, 'Sample_ID', df_rsd_T.index)
            df_rsd_T = df_rsd_T.reset_index(drop=True)
            
            all_signals.append(df_signal_T)
            all_rsds.append(df_rsd_T)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            continue
    
    # Combine all signal dataframes
    if not all_signals or not all_rsds:
        raise ValueError("No valid data was processed from the input files")
    
    # Merge all signals
    combined_signals = pd.concat(all_signals, ignore_index=True)
    
    # Merge all RSDs
    combined_rsds = pd.concat(all_rsds, ignore_index=True)
    
    # Combine signals and RSDs
    final_df = combined_signals.merge(combined_rsds, on='Sample_ID', how='inner')
    
    # Add sample classifications
    final_df['Sample_Type'] = final_df['Sample_ID'].apply(assign_sample_type)
    final_df['Standard_Type'] = final_df['Sample_ID'].apply(assign_standard_type)
    
    return final_df

if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog to select multiple .ASC files
    filenames = filedialog.askopenfilenames(
        title="Select ASC files",
        filetypes=[("ASC files", "*.ASC"), ("All files", "*.*")]
    )
    
    if filenames:
        # Prompt for output directory
        output_dir = filedialog.askdirectory(title="Select the output directory")
        
        if output_dir:
            # Prompt for output filename
            file_csv = simpledialog.askstring("Output File Name", "Enter the filename (without extension):")
            
            if file_csv:
                try:
                    # Process all selected files
                    combined_df = load_asc_files(filenames)
                    
                    # Save combined results
                    output_path = f"{output_dir}/{file_csv}.csv"
                    combined_df.to_csv(output_path, index=None)
                    
                    messagebox.showinfo(
                        "Files Processed",
                        f"Successfully processed {len(filenames)} files.\nCombined data saved as {output_path}"
                    )
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred: {str(e)}")
            else:
                messagebox.showerror("Error", "No file name provided.")
        else:
            messagebox.showerror("Error", "No output directory selected.")
    else:
        messagebox.showerror("Error", "No files selected.")