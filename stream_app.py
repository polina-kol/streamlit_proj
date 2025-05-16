import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Заголовок
st.title("🏠 Предсказание стоимости недвижимости")
st.markdown("Загрузите CSV-файл с данными о домах — сервис автоматически обработает данные и выдаст прогноз цены.")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")

if uploaded_file is not None:
    # Чтение файла
    df = pd.read_csv(uploaded_file)
    st.write("### Данные загружены:")
    st.dataframe(df.head())

    # Проверяем наличие Id
    ids = df['Id'] if 'Id' in df.columns else pd.Series(range(len(df)))

    # Удаляем Id перед обработкой
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Загрузка препроцессора и модели
    try:
        model = joblib.load('voting_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
    except FileNotFoundError:
        st.error("Файл модели или препроцессора не найден. Убедитесь, что они сохранены.")
        st.stop()

    # Применение той же логики предобработки
    try:
        # Повторите те же шаги обработки, что и в train.ipynb
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

        # То же самое создание новых признаков
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['HasRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
        df['TotalQuality'] = df['OverallQual'] * df['OverallCond']
        df['TotalPorchSF'] = df[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].sum(axis=1)
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalSF']

        # Удаление лишних колонок
        cols_to_drop = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath',
                        'BsmtFullBath', 'BsmtHalfBath', 'YrSold', 'YearBuilt', 'YearRemodAdd',
                        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                        'OverallQual', 'OverallCond']
        df.drop([col for col in cols_to_drop if col in df.columns], axis=1, inplace=True)

        # Качество в числах
        quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
            if col in df.columns:
                df[col + '_Num'] = df[col].map(quality_map).fillna(0).astype(int)

        # Удаление слабых признаков
        low_corr_features = ['MoSold', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'KitchenAbvGr',
                             'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF']
        df.drop([col for col in low_corr_features if col in df.columns], axis=1, inplace=True)

        # Прогноз
        predictions_log = model.predict(df)
        predictions = np.expm1(predictions_log)

        # Результат
        result_df = pd.DataFrame({
            'Id': ids.reset_index(drop=True),
            'Predicted_SalePrice': predictions
        })

        st.success("✅ Предсказания успешно выполнены!")
        st.write("### Результаты:")
        st.dataframe(result_df.head(10))

        # Скачивание
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать результаты",
            data=csv,
            file_name='submission.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")