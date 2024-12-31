import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def get_calibration():
    """
    Ask user if they want to produce calibration table.
    """
    response = messagebox.askyesno(
        "Calibration Table!",
        "Do you want to produce calibration curces?\n\n"
        "Yes = You must provide concetration data\n"
        "No"
    )
    return response

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

def calib_dataframe(df_filt, df_c):
    """Combine filtered dataframe with concentration data
    
    Params
    ------
    df_filt : DataFrame
        Filtered pandas dataframe containing analytes that will be merged
    df_c : DataFrame
        Concetration data for each Standard_Type
    
    Returns
    -------
    df_calib : DataFrame
        Combined dataframe
    """
    # Define analyte
    analyte = df_c.columns.to_list()
    
    # Merge dataframes based on Standard_Type
    dmerge = df_filt[df_filt.Standard_Type != 'non_std'][analyte].merge(df_c, 
                on='Standard_Type', how='left', suffixes=('_cps', '_conc'))
    
    # Sort based on Standard_Type
    dmerge = dmerge.sort_values(by='Standard_Type')

    # Get all column names without suffixes
    base_columns = list(set(col.rsplit('_', 1)[0] for col in dmerge.columns if '_cps' in col))

    # Create ordered column list
    ordered_columns = ['Standard_Type']
    for base_col in base_columns:
        ordered_columns.extend([f'{base_col}_cps', f'{base_col}_conc'])
    
    df_calib = dmerge[ordered_columns]
    
    return df_calib

def calculate_blank_averages(filtered_df):
    """
    Calculate average CPS values for blank samples for each analyte
    
    Parameters
    ----------
    filtered_df : pandas.DataFrame
        Filtered DataFrame containing both blank and non-blank samples
        
    Returns
    -------
    pd.Series
        Series containing average blank values for each analyte
    """
    # Get blank samples
    blank_df = filtered_df[filtered_df['Sample_Type'] == 'blank']
    
    if blank_df.empty:
        raise ValueError("No blank samples found in the dataset")
    
    # Get analyte columns (excluding metadata and RSD columns)
    analyte_cols = [col for col in filtered_df.columns 
                   if not any(x in col.lower() for x in ['sample_id', 'unique_id', 'sample_type', 'standard_type', '_rsd'])]
    
    # Calculate mean blank values for each analyte
    blank_averages = blank_df[analyte_cols].mean()
    
    # Calculate standard deviation of blanks for quality control
    blank_stds = blank_df[analyte_cols].std()
    
    # Calculate relative standard deviation of blanks (%)
    blank_rsds = (blank_stds / blank_averages * 100)
    
    # Create a DataFrame with blank statistics
    blank_stats = pd.DataFrame({
        'mean': blank_averages,
        'std': blank_stds,
        'rsd': blank_rsds
    })
    
    return blank_stats

def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

def linreg(x_input, y_input, analyte_name):
    """
    Perform linear regression and return coefficients with analyte information
    
    Parameters
    ----------
    x_input : np.array
        Input x values (typically CPS)
    y_input : np.array
        Input y values (typically concentration)
    analyte_name : str
        Name of the analyte being calibrated
        
    Returns
    -------
    dict
        Dictionary containing slope, intercept, r-squared and analyte name
    """
    mask = ~np.isnan(y_input)
    x = x_input[mask]
    y = y_input[mask]
    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    intercept = coef[1]

    # Predict y
    y_pred = np.polyval(coef, x)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y-y_pred)**2)
    rsq = 1 - (ss_res/ss_tot)

    return {
        'analyte': analyte_name,
        'slope': slope,
        'intercept': intercept,
        'r_squared': rsq
    }

def plot_calibration(dframe, save_location=False, prefix=''):
    """
    Plot calibration curves and return calibration coefficients
    
    Parameters
    ----------
    dframe : pandas.DataFrame
        Calibration dataframe containing CPS and concentration columns
    save_location : str, optional
        Location to save the plot
    prefix : str, optional
        Prefix for the plot filename (e.g., 'blank_corrected_')
    """
    from math import ceil
    
    # Get the list of headers to plot
    header_to_plot = dframe.columns.to_list()[1:]

    # Calculate the number of plots needed
    num_plots = int(len(header_to_plot)/2)

    # Number of rows in subplot
    num_rows = int(ceil(num_plots/3))
    
    # Fig_size
    figsize_x = 3.5 * 3
    figsize_y = 3 * num_rows

    # Initiate figure
    fig, axs = subplots_centered(num_rows, 3, figsize=(figsize_x, figsize_y), nfigs=num_plots)

    # Store calibration coefficients
    calibration_coeffs = []

    for i in range(num_plots):
        j = i*2
        x = dframe[header_to_plot[j]].to_numpy()
        y = dframe[header_to_plot[j+1]].to_numpy()
        
        # Get analyte name from the concentration column (removing '_conc' suffix)
        analyte_name = header_to_plot[j+1].rsplit('_', 1)[0]
        
        # Calculate regression and store coefficients
        coeff_dict = linreg(x, y, analyte_name)
        calibration_coeffs.append(coeff_dict)

        # Plotting code remains the same
        axs[i].scatter(x, y, ec='maroon', fc='None', s=100)
        axs[i].plot(x, coeff_dict['slope']*x + coeff_dict['intercept'], 'k-', zorder=-5)
        axs[i].set_xlabel(header_to_plot[j])
        axs[i].set_ylabel(header_to_plot[j+1])
        axs[i].ticklabel_format(axis='x', style='sci', useMathText=True)
        axs[i].text(0.97, 0.05, 
                   '$Y$ = {:.2e}$X$ + {:.2e}\n$R^2$ = {:.3f}'.format(
                       coeff_dict['slope'], 
                       coeff_dict['intercept'], 
                       coeff_dict['r_squared']),
                   transform=axs[i].transAxes, ha='right', fontsize=10)
        
    plt.tight_layout()

    if save_location is not False:
        save_loc = os.path.splitext(save_location)[0]
        if prefix:
            save_loc = f"{save_loc}_{prefix}"
        save_loc = f"{save_loc}.png"
        plt.savefig(save_loc, dpi=300)
    
    # Convert coefficients to DataFrame
    coeff_df = pd.DataFrame(calibration_coeffs)
    
    return coeff_df

def calculate_concentrations(filtered_df, coeff_df):
    """
    Calculate concentrations using calibration coefficients
    
    Parameters
    ----------
    filtered_df : pandas.DataFrame
        DataFrame containing the filtered CPS data
    coeff_df : pandas.DataFrame
        DataFrame containing calibration coefficients
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with calculated concentrations added
    """
    # Create copy of the input DataFrame
    result_df = filtered_df.copy()
    
    # Add calculated concentrations for each analyte
    for _, row in coeff_df.iterrows():
        analyte = row['analyte']
        slope = row['slope']
        intercept = row['intercept']
        
        # Column name for CPS values
        cps_col = f"{analyte}"
        
        if cps_col in filtered_df.columns:
            # Calculate concentrations
            conc_col = f"{analyte}_calc_conc"
            result_df[conc_col] = (filtered_df[cps_col] * slope) + intercept
    
    return result_df

def calculate_blank_corrected_data(filtered_df, blank_stats):
    """
    Apply blank correction to CPS data
    
    Parameters
    ----------
    filtered_df : pandas.DataFrame
        DataFrame containing the filtered CPS data
    blank_stats : pandas.DataFrame
        DataFrame containing blank statistics
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with blank-corrected CPS values
    """
    corrected_df = filtered_df.copy()
    
    for col in blank_stats.index:
        if col in corrected_df.columns:
            blank_value = blank_stats.loc[col, 'mean']
            corrected_df[col] = (corrected_df[col] - blank_value).clip(lower=0)
            
            # Add blank value for reference
            corrected_df[f"{col}_blank_value"] = blank_value
            
    return corrected_df

def process_calibration(df, df_calib, output_path, prefix=''):
    """
    Process calibration data and generate results
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with CPS values
    df_calib : pandas.DataFrame
        Calibration concentration data
    output_path : str
        Base path for saving files
    prefix : str
        Prefix for file names (e.g., 'original' or 'blank_corrected')
        
    Returns
    -------
    tuple
        (results_df, coeff_df, calib_df)
    """
    # Create calibration merge
    df_calib_merge = calib_dataframe(df, df_calib)
    
    # Get calibration coefficients and plot
    coeff_df = plot_calibration(df_calib_merge, output_path, prefix=prefix)
    
    # Calculate concentrations
    results_df = calculate_concentrations(df, coeff_df)
    
    return results_df, coeff_df, df_calib_merge

def calculate_i_ca_ratios(df, sample_type, blank_corrected=False):
    """
    Calculate I:(Ca+Mg) ratios for iodine calcium samples
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing calculated concentrations
    sample_type : str
        Type of analysis ('total_digest' or 'i_ca')
    blank_corrected : bool
        Whether the input data is blank-corrected (used for output naming only)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added ratio columns
    """
    if sample_type != 'i_ca':
        return df
        
    result_df = df.copy()
    
    # Molecular weights in g/mol
    MW = {
        'I': 126.90,
        'Ca': 40.08,
        'Mg': 24.31
    }
    
    try:
        # Always use _calc_conc suffix for input columns
        suffix = '_calc_conc'
        
        # Convert to μmol/L
        I_umol = df[f'I127(LR){suffix}'] / MW['I']
        Ca43_umol = df[f'Ca43(MR){suffix}'] / MW['Ca']
        Ca44_umol = df[f'Ca44(MR){suffix}'] / MW['Ca']
        Mg_umol = df[f'Mg24(MR){suffix}'] / MW['Mg']
        
        # Add appropriate suffix for output ratio columns
        ratio_suffix = '_blank_corrected' if blank_corrected else ''
        
        # Calculate ratios (μmol/mol)
        result_df[f'I_Ca43+Mg_ratio{ratio_suffix}'] = I_umol / (Ca43_umol + Mg_umol) * 1000
        result_df[f'I_Ca44+Mg_ratio{ratio_suffix}'] = I_umol / (Ca44_umol + Mg_umol) * 1000
        
    except Exception as e:
        print(f"Error calculating ratios: {str(e)}")
        
    return result_df

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

def save_dataframe(df, base_path, excel=False, filtered=False, calib=False, blank=False, 
                  results=False, coeff=False, prefix=None):
    """
    Save the DataFrame with proper prefix handling
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    base_path : str
        Base path for saving files (without extension)
    prefix : str, optional
        Prefix to add before the type suffix (e.g., 'original' or 'blank_corrected')
    """
    # Split the path into base and extension
    path_parts = os.path.splitext(base_path)
    base_without_ext = path_parts[0]
    
    # Build the filename components
    components = [base_without_ext]
    
    # Add prefix if provided
    if prefix:
        components.append(prefix)
    
    # Add type suffix
    if filtered:
        components.append('filtered')
    elif calib:
        components.append('calibration_table')
    elif blank:
        components.append('blank_statistics')
    elif results:
        components.append('results')
    elif coeff:
        components.append('coefficients')
    
    # Join components with underscores
    final_base = '_'.join(components)
    
    # Save as CSV
    csv_path = f"{final_base}.csv"
    df.to_csv(csv_path, index=None)
    
    # Save as Excel if requested
    if excel:
        xlsx_path = f"{final_base}.xlsx"
        df.to_excel(xlsx_path, index=None)
        return csv_path, xlsx_path
    else:
        return csv_path

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
                csv_path = save_dataframe(combined_df, output_path)
                
                # Create and save filtered version
                filtered_df = filter_columns(combined_df, sample_type)
                filtered_csv_path = save_dataframe(
                    filtered_df, output_path, filtered=True
                )
                
                # Get whether user want to supply calibration data
                calibration = get_calibration()
                
                if calibration:
                    calib_file = filedialog.askopenfilename(
                        title="Select the Calibration Table", 
                        filetypes=[("CSV files", "*.csv"), 
                        ("All files", "*.*")]
                    )

                    try:
                        # Calculate blank statistics
                        blank_stats = calculate_blank_averages(filtered_df)
                        blank_stats_path = save_dataframe(
                            blank_stats, output_path, blank=True
                        )
                        
                        # Read calibration data
                        df_calib = pd.read_csv(calib_file)
                        
                        # Process original version (without blank correction)
                        results_df_orig, coeff_df_orig, calib_df_orig = process_calibration(
                            filtered_df, df_calib, output_path, prefix='original'
                        )
                        
                        # Save original version outputs
                        results_csv_path_orig = save_dataframe(
                            results_df_orig, output_path, results=True, prefix='original'
                        )
                        coeff_csv_path_orig = save_dataframe(
                            coeff_df_orig, output_path, coeff=True, prefix='original'
                        )
                        calib_csv_path_orig = save_dataframe(
                            calib_df_orig, output_path, calib=True, prefix='original'
                        )
                        
                        # Process blank-corrected version
                        corrected_df = calculate_blank_corrected_data(filtered_df, blank_stats)
                        results_df_corr, coeff_df_corr, calib_df_corr = process_calibration(
                            corrected_df, df_calib, output_path, prefix='blank_corrected'
                        )
                        
                        # Save blank-corrected version outputs
                        results_csv_path_corr = save_dataframe(
                            results_df_corr, output_path, results=True, prefix='blank_corrected'
                        )
                        coeff_csv_path_corr = save_dataframe(
                            coeff_df_corr, output_path, coeff=True, prefix='blank_corrected'
                        )
                        calib_csv_path_corr = save_dataframe(
                            calib_df_corr, output_path, calib=True, prefix='blank_corrected'
                        )
                        
                        # Calculate I:Ca+Mg ratios if it's an i_ca sample
                        if sample_type == 'i_ca':
                            results_df_orig = calculate_i_ca_ratios(
                                results_df_orig, sample_type, blank_corrected=False
                            )
                            results_df_corr = calculate_i_ca_ratios(
                                results_df_corr, sample_type, blank_corrected=True
                            )
                            
                            # Save updated results with ratios
                            results_csv_path_orig = save_dataframe(
                                results_df_orig, output_path, results=True, prefix='original'
                            )
                            results_csv_path_corr = save_dataframe(
                                results_df_corr, output_path, results=True, prefix='blank_corrected'
                            )

                        messagebox.showinfo(
                            "Files Processed",
                            f"Successfully processed {len(filenames)} files.\n\n"
                            f"Complete data saved as:\n"
                            f"CSV: {os.path.basename(csv_path)}\n"
                            f"Filtered data saved as:\n"
                            f"CSV: {os.path.basename(filtered_csv_path)}\n"
                            f"Blank statistics saved as:\n"
                            f"CSV: {os.path.basename(blank_stats_path)}\n\n"
                            f"Original version files:\n"
                            f"Results: {os.path.basename(results_csv_path_orig)}\n"
                            f"Coefficients: {os.path.basename(coeff_csv_path_orig)}\n"
                            f"Calibration: {os.path.basename(calib_csv_path_orig)}\n\n"
                            f"Blank-corrected version files:\n"
                            f"Results: {os.path.basename(results_csv_path_corr)}\n"
                            f"Coefficients: {os.path.basename(coeff_csv_path_corr)}\n"
                            f"Calibration: {os.path.basename(calib_csv_path_corr)}"
                        )

                    except Exception as e:
                        messagebox.showerror("Error", f"An error occurred during calibration: {str(e)}")
                
                else:
                    messagebox.showinfo(
                        "Processing Complete", 
                        f"Successfully processed {len(filenames)} files.\n\n"
                        f"Complete data saved as:\n"
                        f"CSV: {os.path.basename(csv_path)}\n"
                        f"Filtered data saved as:\n"
                        f"CSV: {os.path.basename(filtered_csv_path)}"
                    )

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
        else:
            messagebox.showerror("Error", "No output path selected.")
    else:
        messagebox.showerror("Error", "No files selected.")
