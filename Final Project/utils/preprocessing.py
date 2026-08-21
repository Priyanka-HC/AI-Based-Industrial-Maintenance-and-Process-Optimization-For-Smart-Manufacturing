import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    # Load dataset
    df = pd.read_csv("dataset/ai4i2020.csv")

    # Remove unnecessary columns
    df.drop(["UDI", "Product ID"], axis=1, inplace=True)

    # Convert Type column into numbers
    df["Type"] = df["Type"].map({
        "L": 0,
        "M": 1,
        "H": 2
    })

    # Features
    X = df.drop("Machine failure", axis=1)

    # Target
    y = df["Machine failure"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler