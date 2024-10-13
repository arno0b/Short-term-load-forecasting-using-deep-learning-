def generate_time_features(df):
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['temp_range'] = df['maxtempC'] - df['mintempC']
    return df

def generate_weekend_feature(df):
    df['is_weekend'] = df.index.weekday >= 5
    return df
