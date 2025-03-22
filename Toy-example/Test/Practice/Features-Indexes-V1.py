import pandas as pd
import matplotlib.pyplot as plt

# Directly assign the file path
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Abnormalities_Features-V4.xlsx"

# Apply LaTeX Formatting for Plots
plt.rcParams.update({'text.usetex': True})  # Enable LaTeX rendering
plt.rcParams.update({'font.family': 'serif'})  # Use serif font
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})  
plt.rcParams.update({'font.size': 6})  # Standard font size
plt.rcParams.update({'mathtext.rm': 'serif'})  # Use serif fonts for math text
plt.rcParams.update({'mathtext.fontset': 'custom'})  # Use custom math fonts

# Load the Excel file
df = pd.read_excel(file_path)

# Clean the dataframe to properly set the header and index
df.columns = df.iloc[0]  # Set the header as the first row
df = df[1:]  # Remove the header row from data
df = df.rename(columns=lambda x: str(x).strip())  # Strip any extra spaces from column names

# Print all column names to understand the issue
print("\nAll Column Names from the DataFrame:")
print(df.columns.tolist())

# Convert the Index column to numeric
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)  # Remove rows where 'Index' could not be converted

# Automatically detect all numerical columns except 'Index'
available_columns = [col for col in df.columns if col != 'Index']
print("\nAvailable columns to plot:", available_columns)

# Prompt the user to enter desired columns
user_input = input("\nEnter the columns you want to plot (comma-separated, or type 'all' to plot all): ")

# Process user input
if user_input.lower() == 'all':
    columns_to_plot = available_columns
else:
    columns_to_plot = [col.strip() for col in user_input.split(',') if col.strip() in available_columns]

# Validate selected columns
if not columns_to_plot:
    print("No valid columns selected. Exiting...")
else:
    print("\nSelected columns to plot:", columns_to_plot)

    # Plot the selected features over the Index
    plt.figure(figsize=(12, 8))

    for col in columns_to_plot:
        try:
            plt.scatter(df['Index'], pd.to_numeric(df[col], errors='coerce'), label=col)
        except Exception as e:
            print(f"Skipping column '{col}' due to error: {e}")

    plt.title(r"Selected Features over Index (-50 to 50)", fontsize=10)
    plt.xlabel(r"Index", fontsize=12)
    plt.ylabel(r"Feature Value", fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True)

    # Show the plot
    plt.show()
