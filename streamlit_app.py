import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузка модели и препроцессора
@st.cache_resource
def load_model():
    model = joblib.load('voting_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# Функция создания новых фичей
def preprocess_input(df):
    df = df.copy()

    # Создание новых фичей (точно как при обучении)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['IsNew'] = (df['HouseAge'] <= 2).astype(int)
    df['HasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['TotalQuality'] = df['OverallQual'] * df['OverallCond']
    df['TotalPorchSF'] = df[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalSF']

    # Ценовое кодирование района
    neighborhood_price = {
        'Blmngtn': 208500.0, 'Blueste': 142000.0, 'BrDale': 106500.0,
        'BrkSide': 137900.0, 'ClearCr': 212500.0, 'CollgCr': 197200.0,
        'Crawfor': 202500.0, 'Edwards': 99950.0, 'Gilbert': 195000.0,
        'IDOTRR': 92500.0, 'MeadowV': 95000.0, 'Mitchel': 159950.0,
        'NAmes': 145000.0, 'NPkVill': 127500.0, 'NWAmes': 175000.0,
        'NoRidge': 260000.0, 'NridgHt': 310000.0, 'OldTown': 125000.0,
        'SWISU': 133000.0, 'Sawyer': 135000.0, 'SawyerW': 181000.0,
        'Somerst': 190000.0, 'StoneBr': 294000.0, 'Timber': 206500.0,
        'Veenker': 241500.0
    }
    df['NeighborhoodPrice'] = df['Neighborhood'].map(neighborhood_price).fillna(163000.0)

    # Логарифмирование
    skewed_features = ['LotArea', 'TotalSF', 'GrLivArea', 'NeighborhoodPrice']
    for feature in skewed_features:
        if feature in df.columns:
            df[feature+'_Log'] = np.log1p(df[feature])

    # Удаление слабо коррелированных признаков
    low_corr_features = ['MoSold', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'KitchenAbvGr',
                         'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF']
    df = df.drop([col for col in low_corr_features if col in df.columns], axis=1)

    # Пропуски
    missing_as_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                       'MasVnrType']
    missing_as_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                       'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

    for feature in df.columns:
        if feature in missing_as_none:
            df[feature] = df[feature].fillna('None')
        elif feature in missing_as_zero:
            df[feature] = df[feature].fillna(0)
        elif df[feature].dtype == 'object':
            df[feature] = df[feature].fillna(df[feature].mode()[0])
        else:
            df[feature] = df[feature].fillna(df[feature].median())
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        if col in df.columns:
            df[col + '_Num'] = df[col].map(quality_map).fillna(0).astype(int)

    return df

# Интерфейс
st.title("🏠 Предсказание стоимости недвижимости")
st.markdown("Загрузите CSV-файл с объектами недвижимости, и я оценю их стоимость.")

uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    processed_data = preprocess_input(data)  # Убедитесь, что функция определена

    if st.button("Предсказать стоимость"):
        try:
            predictions_log = model.predict(processed_data)  # Убедитесь, что модель загружена
            predictions = np.expm1(predictions_log)

            result_df = pd.DataFrame({
                'Id': data['Id'],
                'SalePrice': predictions
            })

            st.success("Предсказания успешно выполнены!")
            st.dataframe(result_df.head())

            # Подготовка к скачиванию
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Скачать результаты",
                data=csv,
                file_name='predicted_prices.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Ошибка при обработке данных: {str(e)}")