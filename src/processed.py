import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    return pd.read_csv(path)


def information(data):
    print("Number of different categories in each column:")
    print(
        f"Countries: {data['country'].nunique()}\n"
        f"Stations: {data['station'].nunique()}\n"
        f"Cities: {data['city'].nunique()}"
    )

    print("\nDataset Info:")
    print(f"Total rows: {data.shape[0]}")
    print(f"Total columns: {data.shape[1]}")

    print("\nDescription:")
    print(data.describe())
    return data  # keep pipeline consistent


def missing_values(data):
    print("\nMissing values:")
    print(data.isnull().sum())
    return data


def feature_engineering(data):
    data["datetime"] = pd.to_datetime(data["last_update"])

    data["day"] = data["datetime"].dt.day
    data["month"] = data["datetime"].dt.month
    data["year"] = data["datetime"].dt.year
    data["hour"] = data["datetime"].dt.hour

    print(data.head())

    return data


def remove_columns(data):
    data = data.drop(columns=["last_update", "station", "country", "datetime"])

    print(data.head())

    return data


def encode_categorical(data):
    encoder = LabelEncoder()

    data["city"] = encoder.fit_transform(data["city"])
    data["state"] = encoder.fit_transform(data["state"])
    data["pollutant_id"] = encoder.fit_transform(data["pollutant_id"])

    print(data.head())

    return data


def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"\nProcessed data saved to {path}")


def run_pipeline(input_path, output_path):
    data = load_data(input_path)

    # Debug guard (prevents silent failure)
    if data is None:
        raise ValueError("Data loading failed")

    data = information(data)
    data = missing_values(data)
    data = feature_engineering(data)
    data = remove_columns(data)
    data = encode_categorical(data)

    print(data.head())  # Final check before saving

    save_data(data, output_path)


if __name__ == "__main__":
    run_pipeline("Dataset/aqi_in_india.csv", "Dataset/clean_air_quality.csv")
