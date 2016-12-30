def subsample(df, lower, upper, step):

    df = df.loc[lower:upper]

    # drop any rows that have missing values
    df = df.dropna()

    # subsample according to step
    df = df.iloc[::step]

    return df
