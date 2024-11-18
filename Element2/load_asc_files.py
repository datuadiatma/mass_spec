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
    if "std_6x" in sample_id.lower(): 
        return "std_6x"
    if "std_6" in sample_id.lower(): 
        return "std_6"
    if "std_7" in sample_id.lower(): 
        return "std_7"
    if "std_8x" in sample_id.lower(): 
        return "std_8x"
    if "std_8" in sample_id.lower(): 
        return "std_8"
    if "std_9" in sample_id.lower(): 
        return "std_9"
    if "std_1" in sample_id.lower(): 
        return "std_1"
    else:
        return "non_std"
        

def get_sample_type():
    """
    Ask user to specify the type of samples being processed.
    Returns either 'total_digest' or 'i_ca'.
    """
    response = messagebox.askyesno(
        "Sample Type",
        "Are these Trace Metal Total Digest samples?\n\n"
        "Yes = Trace Metal Total Digest\n"
        "No = Iodine Calcium"
    )
    return 'total_digest' if response else 'i_ca'

def filter_columns(df, sample_type):
    """
    Filter DataFrame columns based on sample type.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    sample_type : str
        Either 'total_digest' or 'i_ca'
    
    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with only specified columns
    """
    total_digest_cols = [
        'Sample_ID', 'Sample_Type', 'Standard_Type',
        'Ti47(LR)', 'Ti49(LR)', 'V51(LR)', 'Mo95(LR)', 'Mo98(LR)', 
        'In115(LR)', 'U238(LR)', 'Li7(MR)', 'Al27(MR)', 'Mn55(MR)', 
        'Fe56(MR)', 'Fe57(MR)', 'In115(MR)', 'U238(MR)',
        'Ti47(LR)_rsd', 'Ti49(LR)_rsd', 'V51(LR)_rsd', 'Mo95(LR)_rsd', 
        'Mo98(LR)_rsd', 'In115(LR)_rsd', 'U238(LR)_rsd', 'Li7(MR)_rsd', 
        'Al27(MR)_rsd', 'Mn55(MR)_rsd', 'Fe56(MR)_rsd', 'Fe57(MR)_rsd', 
        'In115(MR)_rsd', 'U238(MR)_rsd'
    ]
    
    i_ca_cols = [
        'Sample_ID', 'Sample_Type', 'Standard_Type',
        'I127(LR)', 'Mg24(MR)', 'Ca43(MR)', 'Ca44(MR)',
        'I127(LR)_rsd', 'Mg24(MR)_rsd', 'Ca43(MR)_rsd', 'Ca44(MR)_rsd'
    ]
    
    columns_to_keep = total_digest_cols if sample_type == 'total_digest' else i_ca_cols
    
    # Filter columns that exist in the DataFrame
    existing_cols = [col for col in columns_to_keep if col in df.columns]
    
    return df[existing_cols]

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
    Adds file identifier to sample IDs to ensure uniqueness.
    """
    all_signals = []
    all_rsds = []
    
    for i, file_name in enumerate(file_names, 1):
        try:
            # Get dimensions of the file
            row_num, col_num = pd.read_csv(file_name, delimiter='\t').shape
            
            # Define rows to skip and columns to use
            skip_rows = [1, 2, 3, 4, 5, row_num-3, row_num-2, row_num-1, row_num,
                        row_num+1, row_num+2]
            column = [x for x in range(0, col_num-1) if x % 2 == 1 or x == 1 or x == 0]
            
            # Read the file
            df = pd.read_csv(file_name, delimiter='\t', usecols=column, skiprows=skip_rows)
            
            # Get base filename without extension and path
            base_filename = os.path.splitext(os.path.basename(file_name))[0]
            
            # Separate signal and rsd columns
            sample_name = [col for i, col in enumerate(df.columns) if i%2==1 or i==0]
            rsd = [col for i, col in enumerate(df.columns) if i%2==0 or i==0]
            
            # Process signal data
            df_signal = df[sample_name]
            df_signal_T = df_signal.set_index('Unnamed: 0').T
            
            # Process RSD data
            df_rsd = df[rsd]
            df_rsd = df_rsd.rename(columns=dict(zip(df_rsd.columns, sample_name)))
            df_rsd['Unnamed: 0'] = df_rsd['Unnamed: 0'].apply(lambda x: x + "_rsd")
            df_rsd_T = df_rsd.set_index('Unnamed: 0').T
            
            # Make a copy of index to preserve the original Sample_ID name
            df_signal_T['Original_Sample_ID'] = df_signal_T.index
            
            # Modify index (sample IDs) to include file identifier
            df_signal_T.index = df_signal_T.index + f"_{base_filename}"
            df_rsd_T.index = df_rsd_T.index + f"_{base_filename}"
            
            # Add to lists
            all_signals.append(df_signal_T)
            # all_rsds.append(df_rsd_T_mean)
            all_rsds.append(df_rsd_T)
            
            # Print progress
            print(f"Processed file {i}/{len(file_names)}: {base_filename}")
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            continue
    
    # Combine all signal dataframes
    if not all_signals or not all_rsds:
        raise ValueError("No valid data was processed from the input files")
    
    # Merge all signals and remove duplicates
    combined_signals = pd.concat(all_signals, ignore_index=False)
    
    # Merge all RSDs
    combined_rsds = pd.concat(all_rsds, ignore_index=False)
    
    # Reset indices to columns
    combined_signals.reset_index(inplace=True)
    combined_rsds.reset_index(inplace=True)
    
    # Rename index column to Sample_ID
    combined_signals.rename(columns={'index': 'Unique_ID'}, inplace=True)
    combined_rsds.rename(columns={'index': 'Unique_ID'}, inplace=True)
    
    # Add original Sample_ID column (without file identifier)
    # combined_signals['Sample_ID'] = combined_signals['Unique_ID'].apply(lambda x: x.rsplit('_', 5)[0])
    combined_signals.rename(columns={'Original_Sample_ID': 'Sample_ID'}, inplace=True)
    
    # Combine signals and RSDs
    final_df = combined_signals.merge(combined_rsds, on='Unique_ID', how='inner')
    
    # Add sample classifications (using Original_Sample_ID to maintain correct classification)
    final_df['Sample_Type'] = final_df['Sample_ID'].apply(assign_sample_type)
    final_df['Standard_Type'] = final_df['Sample_ID'].apply(assign_standard_type)
    
    # Reorder columns to put Sample_ID and Original_Sample_ID first
    cols = final_df.columns.tolist()
    cols = ['Unique_ID', 'Sample_ID', 'Sample_Type', 'Standard_Type'] + [col for col in cols if col not in ['Unique_ID', 'Sample_ID', 'Sample_Type', 'Standard_Type']]
    final_df = final_df[cols]
    
    return final_df

def save_dataframe(df, base_path, filtered=False):
    """
    Save the DataFrame in both CSV and Excel formats
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    base_path : str
        Base path for saving files (without extension)
    filtered : bool
        Whether this is a filtered DataFrame (adds _filtered to filename)
    """
    # Remove any extension from the base path
    base_path = os.path.splitext(base_path)[0]
    
    # Add filtered suffix if needed
    if filtered:
        base_path = f"{base_path}_filtered"
    
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
    
    # Get sample type from user
    sample_type = get_sample_type()
    
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
                
                # Save original combined results
                csv_path, xlsx_path = save_dataframe(combined_df, output_path)
                
                # Create and save filtered version
                filtered_df = filter_columns(combined_df, sample_type)
                filtered_csv_path, filtered_xlsx_path = save_dataframe(
                    filtered_df, output_path, filtered=True
                )
                
                messagebox.showinfo(
                    "Files Processed",
                    f"Successfully processed {len(filenames)} files.\n\n"
                    f"Complete data saved as:\n"
                    f"CSV: {os.path.basename(csv_path)}\n"
                    f"Excel: {os.path.basename(xlsx_path)}\n\n"
                    f"Filtered data saved as:\n"
                    f"CSV: {os.path.basename(filtered_csv_path)}\n"
                    f"Excel: {os.path.basename(filtered_xlsx_path)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showerror("Error", "No output path selected.")
    else:
        messagebox.showerror("Error", "No files selected.")