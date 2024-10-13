import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path, outlier_columns):
    # Load dataset
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['date'], utc=True, infer_datetime_format=True)
    df.drop(['date'], axis=1, inplace=True)
    df.set_index('time', inplace=True)

    # Handle outliers
    Q1 = df[outlier_columns].quantile(0.25)
    Q3 = df[outlier_columns].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[outlier_columns] < (Q1 - 1.5 * IQR)) | (df[outlier_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

def generate_time_features(df):
    # Time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['temp_range'] = df['maxtempC'] - df['mintempC']
    return df

def generate_weekend_feature(df):
    # Weekend feature
    df['is_weekend'] = df.index.weekday >= 5
    return df

def split_and_normalize_data(df, target_column='loads', test_size=0.2, random_state=42):
    # Split into features and target
    X = df[df.columns.drop(target_column)].values
    y = df[target_column].values.reshape(-1, 1)

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalization
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)

    return X_train, X_val, y_train, y_val, scaler_X, scaler_y
