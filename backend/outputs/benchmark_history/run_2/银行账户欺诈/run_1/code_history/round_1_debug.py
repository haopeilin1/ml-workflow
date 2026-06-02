if cat_cols:
       encoder = OneHotEncoder(...)
       X_cat_encoded = encoder.fit_transform(X[cat_cols])
       ...
       X = pd.concat([X_num_df, X_cat_df], axis=1)