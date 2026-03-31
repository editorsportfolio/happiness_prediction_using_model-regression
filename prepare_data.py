import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load your prepared CSV
df = pd.read_csv("data.csv")

# Features and target
X = df[["GDP", "Social", "Health", "Freedom"]]
y = df["Happiness"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("happiness_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as happiness_model.pkl")