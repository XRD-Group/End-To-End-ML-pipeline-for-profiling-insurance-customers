import pandas as pd

csv_path = "../data/raw/trainset.csv"


df = pd.read_csv(csv_path)


from sklearn.model_selection import train_test_split

historic_dataset, new_dataset = train_test_split(df, test_size=0.5, random_state=42)

# # Check if the datasets are disjoint
assert len(historic_dataset.merge(new_dataset)) == 0, "Datasets are not disjoint"

# Save the datasets
historic_dataset.to_csv("./data/historic_dataset.csv", index=False)
new_dataset.to_csv("./data/new_dataset.csv", index=False)

# # Randomly shuffle and split the data 50-50
# print("Len df before", len(df))
# historic_dataset = df.sample(frac=.5, random_state=42,replace=False)
# print("Len historic_dataset", len(historic_dataset))
# new_dataset = df.drop(historic_dataset.index).reset_index(drop=True)
# print("Len new_dataset", len(new_dataset))

# # check if the datasets are disjoint
# assert len(historic_dataset.merge(new_dataset)) == 0, "Datasets are not disjoint"
