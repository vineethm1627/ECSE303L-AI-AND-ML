from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def get_column_names_from_ColumnTransformer(column_transformer, orig_cols, verbose=False):
    col_name = []
    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_[:-1]:
        if verbose:
            print('transformer name:', transformer_in_columns[0])

        raw_col_name = list(transformer_in_columns[2])
        transformer = None
        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, OneHotEncoder):
                names = list(transformer.get_feature_names(raw_col_name))
            else:
                names = list(transformer.get_feature_names())

        except AttributeError as error:
            names = raw_col_name
        
        if verbose:
            print("Output column names", names)
        col_name.extend(names)
        
        rem_names = [orig_cols[c] for c in column_transformer.transformers_[-1][2]]
        col_name.extend(rem_names)

    return col_name