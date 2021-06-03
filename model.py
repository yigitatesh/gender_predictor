# Import Libraries
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression


# Load Data

df = pd.read_csv("weight_height.csv")


# Prepare Data

# take a part of whole data
indexes = np.random.choice(np.array(df.index), size=3000)
df = df.iloc[indexes, :]

X = df.drop("Gender", axis=1).values
y = df["Gender"].values

# label encoding

encoder = LabelEncoder().fit(y)
y = encoder.transform(y)

index2label = {0: "Female", 1: "Male"}


# Create Model

lr = LogisticRegression(max_iter=1000).fit(X, y)


# Save Model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


# Load Model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Test Model

pred = model.predict(np.array([[175, 80]]))

print("Predicted as {}".format(index2label[pred[0]]))