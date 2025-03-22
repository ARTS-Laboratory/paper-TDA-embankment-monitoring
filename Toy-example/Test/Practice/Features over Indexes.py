import pandas as pd
import matplotlib.pyplot as plt
import re

# file path
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Abnormalities_Features-V4.xlsx"

#  LaTeX Formatting for Plots
plt.rcParams.update({'text.usetex': True})  
plt.rcParams.update({'font.family': 'serif'})  
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 6})  
plt.rcParams.update({'mathtext.rm': 'serif'})  
plt.rcParams.update({'mathtext.fontset': 'custom'})  

# load the Excel file
df = pd.read_excel(file_path)

# clean the dataframe to properly set the header and index
df.columns = df.iloc[0]  # Set the header as the first row
df = df[1:]  # Remove the header row from data

# removing the special characters from column names
df.columns = [re.sub(r'\s+', ' ', str(x).strip()) for x in df.columns]

#  all column names to understand the issue
print("\nAll Column Names from the DataFrame:")
print(df.columns.tolist())

#  index column to numeric
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)  # Remove rows where 'Index' could not be converted

# Automatically detect all numerical columns except 'Index'
available_columns = [col for col in df.columns if col != 'Index']
print("\nAvailable columns to plot:", available_columns)

# specifiyng the desired columns in the code
columns_to_plot = ['PE (H0)', 'PE (H1)', 'Landscape (H0)', 'Landscape (H1)',
                   'Persistence Image (H0)', 'Persistence Image (H1)', 
                   'Betti (H0)', 'Betti (H1)', 'Heat (H0)', 'Heat (H1)']

# validate selected columns after cleaning
columns_to_plot = [col for col in columns_to_plot if col in available_columns]
if not columns_to_plot:
    print("No valid columns selected. Exiting...")
else:
    print("\nSelected columns to plot:", columns_to_plot)

    
    plt.figure(figsize=(6.5, 4), dpi=300)  

    for col in columns_to_plot:
        try:
            plt.scatter(df['Index'], pd.to_numeric(df[col], errors='coerce'), label=col)
        except Exception as e:
            print(f"Skipping column '{col}' due to error: {e}")

  
    plt.xlabel(r"Index", fontsize=10)
    plt.ylabel(r"Feature Value", fontsize=10)
    plt.legend(fontsize=6)
    plt.grid(True)

    # Show the plot
    plt.show()
