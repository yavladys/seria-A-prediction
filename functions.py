import pandas as pd


def get_dummies(df, columns_names, keep_original=False, prefix_name='ht'):
    df_dummies = pd.get_dummies(df[columns_names], columns=columns_names, prefix=prefix_name)
    df = pd.concat([df, df_dummies], axis=1)
    if keep_original:
        return df
    else:
        return df.drop(columns=columns_names)
