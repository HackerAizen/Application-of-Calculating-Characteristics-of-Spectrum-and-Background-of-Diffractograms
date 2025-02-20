import polars as pl

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def calculate_baseline(df: pl.DataFrame, degree: int=2):
    df_pd = df.to_pandas()

    # Установка порядка полинома для Baseline
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(df_pd[['x']])

    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_poly, df_pd['y'])

    # Получаем предсказания (Baseline)
    baseline = model.predict(X_poly)

    # Добавляем Baseline к DataFrame
    df_pd['Baseline'] = baseline

    return df_pd