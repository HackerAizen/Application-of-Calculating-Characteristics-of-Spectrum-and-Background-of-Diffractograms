import polars as pl

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression

from scipy.integrate import simpson

def calculate_baseline(df: pl.DataFrame, degree: int=2, iters: int=100):
    df_pd = df.to_pandas()

    # Установка порядка полинома для Baseline
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(df_pd[['x']])

    # Обучение модели линейной регрессии
    model = RANSACRegressor(LinearRegression(), max_trials=iters)
    model.fit(X_poly, df_pd['y'])

    # Получаем предсказания (Baseline)
    baseline = model.predict(X_poly)

    # Добавляем Baseline к DataFrame
    df_pd['Baseline'] = baseline

    return df_pd

def calculate_peak_areas(Base: pl.DataFrame, PeaksFrame: pl.DataFrame, wind_size: int=0) -> pl.DataFrame:
    # Создаем список для хранения результатов
    # res_df = pl.DataFrame()

    # for rowba in PeaksFrame.iter_rows(named=True):
    #     region_begin = rowba['region begin']
    #     region_end = rowba['region end']
    #     print(region_begin, region_end)
    #     # baseline_df = Base[max(0, region_begin + wind_size):min(region_end - wind_size, Base.shape[0])]
    #     # print(Base.shape)
    #     baseline_df = pl.from_pandas(calculate_baseline(
    #     Base[max(0, region_begin + wind_size):min(region_end - wind_size, Base.shape[0])],
    #     degree=polinom.value) 
    # )

    # # Добавляем результаты в результирующий датафрейм
    #     res_df = pl.concat([res_df, baseline_df])
    # res_df = res_df.with_columns((pl.col('Baseline') * 2).cast(pl.Float64).alias("Baseline 1/2*h"))
    peak_areas = []
    igh = 0

    # Проходим по каждому пику в PeaksFrame
    for peak in PeaksFrame.iter_rows(named=True):
        global_peak_x = peak['global_peak_x']

        start_idx = peak['region begin'] + wind_size
        end_idx = peak['region end'] - wind_size
        differ = end_idx - start_idx
        # print(start_idx, end_idx, differ)

        peak_data = Base[start_idx:end_idx]
        # igh += differ
        # peak_data = Base[differ*igh:differ*(igh+1)]
        igh += 1

        # print(peak_data)

        # Вычисляем площадь под кривой с помощью simpson или trapz
        x_values = peak_data["x"].to_numpy()
        y_values = peak_data["y"].to_numpy()
        baseline_values = peak_data["Baseline"].to_numpy()
        h2_baseline_values = (baseline_values + y_values) / 2

        # Вычитаем базовую линию из y
        corrected_y = y_values - baseline_values
        cory = y_values - h2_baseline_values
        # print(corrected_y)
        # print(cory)
        h2_x_values = x_values[cory >=0]
        x_values = x_values[corrected_y >=0]
        cory = cory[cory >= 0]
        corrected_y = corrected_y[corrected_y >=0]
        # print(len(corrected_y), len(x_values))

        # Вычисляем площадь
        area = simpson(corrected_y, x=x_values)
        # area = np.trapz(corrected_y, x_values)

        # print(cory)
        # print(area_h)
        # if cory.any() and h2_x_values.any():
        #     area_h = simpson(cory, x=h2_x_values)
        area_h = simpson(cory, x=h2_x_values)
        # area_h = np.trapz(cory, h2_x_values)
        # area_2 = np.trapz(corrected_y, x_values)
        # print(area_h)

        # Добавляем результат в список
        peak_areas.append((start_idx, end_idx, area, area_h))

    # Создаем новый DataFrame с результатами
    result_df = pl.DataFrame({
        "start_x": [x[0] for x in peak_areas],
        "end_x": [x[1] for x in peak_areas],
        "peak_area": [x[2] for x in peak_areas],
        "peak_area_h2": [x[3] for x in peak_areas]
    })

    return result_df