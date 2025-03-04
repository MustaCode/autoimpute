import pandas as pd
import numpy as np
import time
import csv
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype, is_string_dtype, is_float_dtype

def imputex(df, df_bench):
    # for col in df:
    #     df[col] = df[col].apply(lambda x: str(x).replace(",", " -"))
    #     df[col]

    df = df.replace(['-'], np.nan)
    df.replace(',',';', regex=True, inplace=True)

    
    # df = df.replace(',', '.')
 # PARSE & INITIAL GUESS SECTION
    print(df.isna().sum().sum(), " Total missing values before imputation")

    # df.to_csv('exports/data.csv', encoding='utf-8')

    start = time.time()
    le = LabelEncoder()

    df_parsed = df.copy()
    df_inim = df.copy()

    # for column in df_parsed:
    #     df_ischar = df[column].str.contains('_|[^.\w\s*]+')
    #     if df_ischar.all():
    #         df_parsed[column] = df_parsed[column].convert_dtypes()
    #         df_inim[column] = df_inim[column].convert_dtypes()
    #         df_parsed[column] = le.fit_transform(df_parsed[column])
    #         continue
            
    #     df_isnumber = df_parsed[column].str.contains('[-+]?\d*\.\d+|\d+')
    #     if df_isnumber.all():
    #         df_parsed[column] = pd.to_numeric(df_parsed[column], errors='coerce')
    #         df_inim[column] = pd.to_numeric(df_parsed[column], errors='coerce')
    #         if df_parsed[column].isna().sum() > 0:
    #             df_parsed[column].interpolate(method='linear', limit_direction = 'both', inplace=True) # Impute MV using linear interpolation
    #             df_inim[column].interpolate(method='linear', limit_direction = 'both', inplace=True)
    #     else:
    #         if df_parsed[column].isna().sum() > 0:
    #             df_parsed[column] = df_parsed[column].fillna(df[column].mode()[0])
    #             df_inim[column] = df_inim[column].fillna(df[column].mode()[0])
    #         df_parsed[column] = df_parsed[column].convert_dtypes()
    #         df_inim[column] = df_inim[column].convert_dtypes()
    #         df_parsed[column] = le.fit_transform(df_parsed[column])
    
    # df_inim = df_inim.iloc[:100, :5]

    # Remove outliers
    for col in df_inim:
        if is_numeric_dtype(df_inim[col]):
            df_inim[(np.abs(stats.zscore(df_inim[col])) < 3)]

    df_inim_complete = df_inim.copy()

    # Save a complete set for fast imputation

    for column in df_inim_complete:
        if is_numeric_dtype(df_inim_complete[column]) and len(df_inim_complete[column].value_counts()) > 9:
            if df_inim_complete[column].isna().sum() > 0:
                df_inim_complete[column].interpolate(method='linear', limit_direction = 'both', inplace=True) # Impute MV using linear interpolation
        else:
            if df_inim_complete[column].isna().sum() > 0:
                df_inim_complete[column] = df_inim_complete[column].fillna(df[column].mode()[0])

    # Start the iteration
    while df_inim.isna().sum().sum() > 0:
        # Select feature with the least MV's as the DV
        mv_col_sorted = df_inim.isna().sum()
        mv_col_sorted.sort_values(ascending=True, inplace=True)
        dv_feature = 0
        counter = 0
        for mv in mv_col_sorted:
            if mv > 0 and dv_feature == 0:
                dv_feature = mv_col_sorted.keys()[counter]
            counter = counter + 1

        # Copy complete IV from the complete dataset
        df_inim_ready = df_inim.copy()
    
        for col in df_inim_ready:
            if df_inim_ready[col].isna().sum() > 0 and col != dv_feature:
                df_inim_ready.loc[df_inim_ready.index, col] = df_inim_complete[col]
        
        # Encode IV categorical features
        for column in df_inim_ready:
            if is_string_dtype(df_inim_ready[column]) and column != dv_feature:
                # SPECIAL CHARACTER SECTION
                df_ischar = df_inim_ready[column].str.contains('_|[^\w\s*]+')
                if df_ischar.all():
                    df_inim_ready[column] = le.fit_transform(df_inim[column])
                    continue
                #df_inim_ready = pd.get_dummies(df_inim_ready, columns=[column], prefix=column, drop_first=True)
                if len(df_inim_ready[column].value_counts()) > 2:       
                    # Apply Hot Encoding
                    df_inim_ready = pd.get_dummies(df_inim_ready, columns=[column], prefix=column, drop_first=True)
                else:
                    # Apply Label Encoding
                    df_inim_ready[column] = le.fit_transform(df_inim_ready[column])
            elif len(df_inim_ready[column].value_counts()) < 10 and column != dv_feature:
                # Apply Hot Encoding
                #df_inim_ready = df_inim_ready.astype({column: int})
                df_inim_ready[column] = pd.to_numeric(df_inim_ready[column], errors='coerce').fillna(0).astype(int)
                df_inim_ready = pd.get_dummies(df_inim_ready, columns=[column], prefix=column, drop_first=True)

        # Take a copy of a numerical dataset with complete X and incomplete y
        X_inim_ready = df_inim_ready.copy()

        # Drop MV instances from DV
        df_inim_ready.dropna(axis=0, how='any', subset=[dv_feature], inplace=True)
        df_inim_ready = df_inim_ready.reset_index(drop=True)

        df_inim_encoded = df_inim_ready.copy()

        dv_type = ''

        if is_float_dtype(df_inim_encoded[dv_feature]) and len(df_inim_encoded[dv_feature].value_counts()) > 9:
            dv_type = 'regression'
        else:
            df_inim_encoded[dv_feature] = le.fit_transform(df_inim_encoded[dv_feature])
            dv_type = 'classification'

        # Split the dataset
        df_split = df_inim_encoded.copy()

        dv = df_split.columns[0]

        # If DV is not in the first column, make it the first column
        if dv != dv_feature:
            df_split.insert(0, dv_feature + '_DV', df_split[dv_feature])
            df_split.drop(dv_feature, axis=1, inplace=True)
            dv_feature_train = dv_feature + '_DV'
            y=df_split[dv_feature_train]
        else:
            y=df_split[dv_feature]

        X = df_split.iloc[:, 1:]

        # Split into training and test set
        if dv_type == 'classification':
            strat_flag = False
            for i in range(len(y.value_counts())):
                if y.value_counts()[i] == 1:
                    strat_flag = True
            if strat_flag == True:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
                    
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
        
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        # Train Model (ExtraTrees & MLP)
        if dv_type == 'regression':
            model = ExtraTreesRegressor(n_estimators=100, random_state=0, max_features="sqrt").fit(X_train, y_train)
        else:
            model = ExtraTreesClassifier(n_estimators=100, random_state=0, max_features="sqrt").fit(X_train, y_train)

        # Impute missing values
        # Impute all IV's that have MV's
        df_split_mv = X_inim_ready.copy()

        dv = df_split_mv.columns[0]

        dv_feature_final = ''

        # If DV is not in the first column, make it the first column
        if dv != dv_feature:
            df_split_mv.insert(0, dv_feature + '_DV', df_split_mv[dv_feature])
            df_split_mv.drop(dv_feature, axis=1, inplace=True)
            dv_feature_mv = dv_feature + '_DV'
            dv_feature_final = dv_feature_mv
            null_rows = df_split_mv[dv_feature_mv].isnull()
        else:
            dv_feature_final = dv_feature
            null_rows = df_split_mv[dv_feature].isnull() # Select a Feature with Missing Values

        df_dv_mv = df_split_mv[null_rows].copy() # prints only those rows where null_filter is True

        # Split IV from DV
        X_mv = df_dv_mv.iloc[:, 1:]

        X_mv = StandardScaler().fit_transform(X_mv)

        #Predict MV's in DV
        dv_mv_predictions = model.predict(X_mv)

        df_dv_mv_predictions = pd.DataFrame(dv_mv_predictions)

        # To convert numeric back to string for categorical feature
        if dv_type == 'classification':
            #lb = LabelBinarizer()
            # dv_labels = df_inim[dv_feature].value_counts().index.values
            arr_dv_mv_predictions = np.array(df_dv_mv_predictions)
            #np.ravel(arr_dv_mv_predictions)
            #arr_dv_mv_predictions = arr_dv_mv_predictions.reshape(-1, 1)

            # predictions_fit = le.fit(dv_labels)
            predictions_ordinal = le.inverse_transform(np.ravel(arr_dv_mv_predictions))
            predictions_ordinal_df = pd.DataFrame(predictions_ordinal)
            df_dv_mv_predictions = predictions_ordinal_df.copy()

        # Replace the MV's in the subset df_split_mv[dv_feature].isnull()
        df_dv_imputed = df_dv_mv.copy()

        df_dv_mv_predictions.index = df_dv_imputed.index
        df_dv_imputed[dv_feature_final] = df_dv_mv_predictions[0]

        # Copy the index of imputed instances from the subset df_split_mv[dv_feature].isnull() to the original dataset
        df_inim.loc[df_dv_imputed.index, dv_feature] = df_dv_imputed[dv_feature_final]

        print('DV Feature: ' + dv_feature)
        print('DV Final: ' + dv_feature)
        print('Model: ' + dv_type)
        print('A feature with missing values has been imputed')

    end = time.time()
    extra_time = end - start
    print(extra_time, "ImputeX in seconds")
    print("==========================")
    print("==========================")
    print("length: ", len(df))


    # PERFORMANCE METRICS
    # df_inim_mv_cp = df_inim.copy()

    # # PARSE TO NUMERIC
    # for column in df_inim_mv_cp:
    #     if is_string_dtype(df_inim_mv_cp[column]):
    #         df_inim_mv_cp[column] = le.fit_transform(df_inim_mv_cp[column])

    # df_inim_mv_cp_arr = np.array(df_inim_mv_cp)

    # df_parsed_cp = df_bench.copy()
    # df_parsed_cp_arr = np.array(df_parsed_cp)

    # for column in df_parsed_cp:
    #     df_parsed_cp[column] = pd.to_numeric(df_parsed_cp[column], errors='coerce')

    # avg_extra_cat_pfc = np.array([])
    # avg_extra_cat_nrmse = np.array([])

    # for col in df_inim_mv_cp:
    #     #if col_index == 0 or col_index == 2 or col_index == 4 or col_index == 5 or col_index == 6:
    #     if col == 'Diagnosis' or col == 'PTGENDER' or col == 'PTETHCAT' or col == 'PTRACCAT' or col == 'PTMARRY':
    #         extra_pfc = 1 - metrics.accuracy_score(df_parsed_cp[col], df_inim_mv_cp[col])
    #         avg_extra_cat_pfc = np.append(avg_extra_cat_pfc, [extra_pfc])
    #     else:
    #         extra_rmse = np.sqrt(mean_squared_error(df_parsed_cp[col], df_inim_mv_cp[col]))
    #         extra_nrmse = extra_rmse / (df_parsed_cp[col].max() - df_parsed_cp[col].min())
    #         avg_extra_cat_nrmse = np.append(avg_extra_cat_nrmse, [extra_nrmse])

    # print('IMPUTEX PFC: ', avg_extra_cat_pfc.mean())
    # print('IMPUTEX NRMSE: ', avg_extra_cat_nrmse.mean())

    print(df_inim.isna().sum().sum(), " Total missing values after imputation")

    # print('ORIG: ', df_parsed_cp.info())
    # print('MISS: ', df_inim_mv_cp.info())
    
    # print('IMP: ', df_inim_mv_cp.head(30))

    # EXPORT THE IMPUTED DATASET
    # df_inim.to_csv('exports/data.csv', encoding='utf-8')
    return df_inim