import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('estatements.csv', delimiter=',')
print(df.columns)
# Ensure the 'Amount($)' column is treated as numeric (float). If it contains non-numeric values or symbols, they should be cleaned or handled.
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Calculate the total amount
total_amount = df['Amount'].sum()

print(f"Total Amount: ${total_amount:.2f}")
