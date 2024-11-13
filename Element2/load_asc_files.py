import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

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

def select_files_from_multiple_folders():
    """
    Allow users to select files from multiple folders and build a cumulative list.
    Returns a list of selected file paths.
    """
    selected_files = []
    
    while True:
        # Get files from current folder
        new_files = filedialog.askopenfilenames(
            title=f"Select ASC files (Current total: {len(selected_files)})\nPress Cancel when done",
            filetypes=[("ASC files", "*.ASC"), ("All files", "*.*")]
        )
        
        if not new_files:  # If Cancel is pressed (no files selected)
            if selected_files:  # If we already have some files
                return selected_files
            else:  # If no files were selected at all
                if messagebox.askyesno("No Files", "No files selected. Do you want to try again?"):
                    continue
                else:
                    return None
        
        # Add new files to our list
        selected_files.extend(new_files)
        
        # Show current selection and ask whether to continue
        files_message = "\n".join([os.path.basename(f) for f in new_files])
        should_continue = messagebox.askyesno(
            "Continue?",
            f"Added {len(new_files)} files:\n{files_message}\n\n"
            f"Total files selected: {len(selected_files)}\n\n"
            "Do you want to select files from another folder?"
        )
        
        if not should_continue:
            return selected_files

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

def save_dataframe(df, base_path):
    """
    Save the DataFrame in both CSV and Excel formats
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    base_path : str
        Base path for saving files (without extension)
    """
    # Remove any extension from the base path
    base_path = os.path.splitext(base_path)[0]
    
    # Save as CSV
    csv_path = f"{base_path}.csv"
    df.to_csv(csv_path, index=None)
    
    # Save as Excel
    xlsx_path = f"{base_path}.xlsx"
    df.to_excel(xlsx_path, index=None)
    
    return csv_path, xlsx_path

if __name__ == "__main__":
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()
    
    # Get files from multiple folders
    filenames = select_files_from_multiple_folders()
    
    if filenames:
        # Use asksaveasfilename to get both directory and filename
        output_path = filedialog.asksaveasfilename(
            title="Save combined data as...",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if output_path:
            try:
                # Process all selected files
                combined_df = load_asc_files(filenames)
                
                # Save combined results in both formats
                csv_path, xlsx_path = save_dataframe(combined_df, output_path)
                
                messagebox.showinfo(
                    "Files Processed",
                    f"Successfully processed {len(filenames)} files.\n"
                    f"Data saved as:\n"
                    f"CSV: {os.path.basename(csv_path)}\n"
                    f"Excel: {os.path.basename(xlsx_path)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showerror("Error", "No output path selected.")
    else:
        messagebox.showerror("Error", "No files selected.")