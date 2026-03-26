from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from preprocessing import load_data, preprocess_data

# Load data
df = load_data("../data/crop_data.csv")
df = preprocess_data(df)

# Features and target
X = df.drop("yield", axis=1)
y = df["yield"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/model.pkl")

print("Model trained and saved.")
