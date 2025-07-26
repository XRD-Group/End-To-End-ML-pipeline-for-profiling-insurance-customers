import joblib

from policyML.bronze.bronze import get_db_connection
import pandas as pd

from pathlib import Path

# set path to the root of the project
path = Path(__file__).parents[2]  

# Import the trained random forest
rf = joblib.load("models/random_forest_model.joblib")

conn = get_db_connection()
schema = 'silver'
table_name = 'insurance'

query = f'SELECT * FROM "{schema}"."{table_name}";'
data = pd.read_sql(query, conn)
conn.close()

#if "amt_claim" in data.columns:
 #   raise ValueError("amt_claim already present in the dataframe")

data.drop(columns=["amt_claim","dwh_create_data","id"],inplace=True)

# Run the inference step on the data
amt_claim_pred = rf.predict(data)

#