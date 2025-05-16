import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# %% Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_IDs = test['Id']
y_train = train['SalePrice']
train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features = test.drop(['Id'], axis=1)

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

# Логарифмируем целевую переменную
y_train_log = np.log1p(y_train)

# %% Пропуски
missing_as_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                  'MasVnrType']
missing_as_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for feature in all_features.columns:
    if feature in missing_as_none:
        all_features[feature] = all_features[feature].fillna('None')
    elif feature in missing_as_zero:
        all_features[feature] = all_features[feature].fillna(0)
    elif all_features[feature].dtype == 'object':
        all_features[feature] = all_features[feature].fillna(all_features[feature].mode()[0])
    else:
        all_features[feature] = all_features[feature].fillna(all_features[feature].median())

# %% Новые признаки
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['TotalBathrooms'] = all_features['FullBath'] + 0.5 * all_features['HalfBath'] + \
                                 all_features['BsmtFullBath'] + 0.5 * all_features['BsmtHalfBath']
all_features['HouseAge'] = all_features['YrSold'] - all_features['YearBuilt']
all_features['RemodAge'] = all_features['YrSold'] - all_features['YearRemodAdd']
all_features['IsNew'] = (all_features['HouseAge'] <= 2).astype(int)
all_features['HasRemodeled'] = (all_features['YearRemodAdd'] != all_features['YearBuilt']).astype(int)
all_features['TotalQuality'] = all_features['OverallQual'] * all_features['OverallCond']
all_features['TotalPorchSF'] = all_features[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
all_features['HasGarage'] = (all_features['GarageArea'] > 0).astype(int)
all_features['HasPool'] = (all_features['PoolArea'] > 0).astype(int)
all_features['HasBsmt'] = (all_features['TotalBsmtSF'] > 0).astype(int)
all_features['HasFireplace'] = (all_features['Fireplaces'] > 0).astype(int)
all_features['LivingAreaRatio'] = all_features['GrLivArea'] / all_features['TotalSF']

# Удаление исходных признаков
cols_to_drop = [
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
    'YrSold', 'YearBuilt', 'YearRemodAdd',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
    'OverallQual', 'OverallCond'
]
all_features.drop([col for col in cols_to_drop if col in all_features.columns], axis=1, inplace=True)

# Качество дома в числах
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
            'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    if col in all_features.columns:
        all_features[col + '_Num'] = all_features[col].map(quality_map).fillna(0).astype(int)

# Ценовое кодирование района
neighborhood_price = train.groupby('Neighborhood')['SalePrice'].median()
all_features['NeighborhoodPrice'] = all_features['Neighborhood'].map(neighborhood_price)
all_features['NeighborhoodPrice'] = all_features['NeighborhoodPrice'].fillna(neighborhood_price.median())

# Логарифмирование
skewed_features = ['LotArea', 'TotalSF', 'GrLivArea', 'NeighborhoodPrice']
for feature in skewed_features:
    if feature in all_features.columns:
        all_features[feature+'_Log'] = np.log1p(all_features[feature])

# Удаление слабо коррелированных признаков
low_corr_features = ['MoSold', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'KitchenAbvGr',
                     'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF']
all_features.drop([col for col in low_corr_features if col in all_features.columns], axis=1, inplace=True)

# Разделение обратно на train и test
X_train = all_features.iloc[:len(train_features)]
X_test = all_features.iloc[len(train_features):]

# Определение числовых и категориальных признаков
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Создание препроцессора
numeric_transformer = Pipeline(steps=[
    ('scaler', RobustScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# VotingRegressor
voting_regressor = VotingRegressor([
    ('lasso', Lasso(alpha=0.0005, random_state=42)),
    ('ridge', Ridge(alpha=10.0, random_state=42)),
    ('elastic', ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                      max_depth=4, max_features='sqrt',
                                      min_samples_leaf=15, min_samples_split=10,
                                      loss='huber', random_state=42)),
    ('xgb', XGBRegressor(objective='reg:squarederror', learning_rate=0.01, n_estimators=3000,
                         max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                         gamma=0.0, reg_alpha=0.5, reg_lambda=1.0, random_state=42)),
    ('forest', RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)),
    ('catboost', CatBoostRegressor(iterations=3000, learning_rate=0.01, depth=6, l2_leaf_reg=3,
                                   border_count=20, loss_function='RMSE', verbose=0, random_state=42)),
    ('Gradient', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=10,
                                          loss='huber', random_state=42)),
    ('lgbm', LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.01, n_estimators=3000,
                           max_bin=255, bagging_fraction=0.8, bagging_freq=5,
                           feature_fraction=0.7, feature_fraction_seed=42,
                           min_sum_hessian_in_leaf=11, random_state=42))
], weights=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# Финальный пайплайн
voting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', voting_regressor)
])

# Обучение модели
print("Обучение VotingRegressor...")
voting_pipeline.fit(X_train, y_train_log)

# Сохранение модели и препроцессора
import joblib
joblib.dump(voting_pipeline, 'voting_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Модель и препроцессор сохранены.")