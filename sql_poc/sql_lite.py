import pandas as pd
from sqlalchemy import create_engine

# Load CSV file into a DataFrame
df = pd.read_csvdf = pd.read_csv('estatements.csv', delimiter=',')

# Create an SQLite in-memory database
engine = create_engine('sqlite://', echo=False)

# Strip any leading/trailing spaces from Spend_Categories
for col in df.columns:
    if col != "Amount":
        df[col] = df[col].str.strip()

# Load DataFrame into the SQLite database
df.to_sql('ESTAT', con=engine, index=False, if_exists='replace')

# Execute SQL queries
query = "SELECT * FROM ESTAT WHERE Spend_Categories = 'Restaurants';"
result = pd.read_sql(query, con=engine)

# Display the result
print(result)