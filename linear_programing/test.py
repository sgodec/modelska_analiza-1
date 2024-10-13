import pandas as pd
df = pd.read_csv('zivila_2.csv')


# Check for any NaN values in Valine_g
nan_valine = df[df['Valine_g'].isna()]
print("Rows with NaN in Valine_g:")
print(nan_valine)

# Optionally, fill NaN with 0.00
df['Valine_g'] = df['Valine_g'].fillna(0.00)

# Save the cleaned CSV
df.to_csv('zivila_2.csv', index=False)
