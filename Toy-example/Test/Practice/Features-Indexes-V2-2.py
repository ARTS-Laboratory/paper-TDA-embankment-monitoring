import pandas as pd
import matplotlib.pyplot as plt
import re

# Directly assign the file path
file_path = "C:/Users/golzardm/Documents/paper-TDA-embankment-monitoring/Toy-example/Data/Abnormalities_Features-V4.xlsx"

# Disable LaTeX rendering explicitly to avoid PDF saving issues
plt.rcParams.update({
    "text.usetex": False,           # Disable LaTeX formatting
    "font.family": "DejaVu Sans",   # Use a compatible font family
    "mathtext.fontset": "dejavusans" # Use DejaVu Sans for math text
})

# Load the Excel file
df = pd.read_excel(file_path)

# Clean the dataframe to properly set the header and index
df.columns = df.iloc[0]  # Set the header as the first row
df = df[1:]  # Remove the header row from data

# Strip spaces and remove special characters from column names
df.columns = [re.sub(r'\s+', ' ', str(x).strip()) for x in df.columns]

# Print all column names to understand the issue
print("\nAll Column Names from the DataFrame:")
print(df.columns.tolist())

# Convert the Index column to numeric
df['Index'] = pd.to_numeric(df['Index'], errors='coerce')
df.dropna(subset=['Index'], inplace=True)  # Remove rows where 'Index' could not be converted

# Automatically detect all numerical columns except 'Index'
available_columns = [col for col in df.columns if col != 'Index']
print("\nAvailable columns to plot:", available_columns)

# Directly specify the desired columns in the code
columns_to_plot = ['PE (H0)', 'PE (H1)', 'NoP (H0)', 'NoP (H1)', 
                   'Bottleneck (H0)', 'Bottleneck (H1)', 'Wasserstein (H0)', 'Wasserstein (H1)',
                   'Landscape (H0)', 'Landscape (H1)']

# Validate selected columns after cleaning
columns_to_plot = [col for col in columns_to_plot if col in available_columns]
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

    plt.title("Selected Features over Index (-50 to 50)", fontsize=10)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Feature Value", fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
    print("Plot saved as plot.pdf")

    # Show the plot
    plt.show()
