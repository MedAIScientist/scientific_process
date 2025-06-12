import pandas as pd

# Input and output file paths
input_file = 'diabetes_binary_health_indicators_BRFSS2015.csv'
output_file = 'diabetes_first_10000_rows.csv'

# Read the first 10,000 rows only
df = pd.read_csv(input_file, nrows=10000)

# Save to a new CSV
df.to_csv(output_file, index=False)

print(f"Saved first . rows to {output_file}")
