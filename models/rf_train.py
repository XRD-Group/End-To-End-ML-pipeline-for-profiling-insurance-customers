import joblib

from policyML.bronze.bronze import get_db_connection
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

# set path to the root of the project
path = Path(__file__).parents[2]  

# Connect to db
conn = get_db_connection()
schema = 'silver'
table_name = 'insurance'

query = f'SELECT * FROM "{schema}"."{table_name}";'
data = pd.read_sql(query, conn)
conn.close()

# drop the ID
data.drop(columns=["id",'dwh_create_data'],inplace=True)


# Train test split
X = data.drop(columns="amt_claim")
y = data["amt_claim"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

# Define ML model
rf = RandomForestRegressor(verbose=10,n_jobs=-1)

rf.fit(X_train,y_train)


# Save the model to a file
joblib.dump(rf, './models/random_forest_model.joblib')