import numpy as np
import polars as pl
from scipy.signal import savgol_filter

def smooth_and_filter_peaks(df, window_size=11, polyorder=2, threshold=0.05):
    # Преобразуем y-значения в numpy массив
    row_h = df['y'].rolling_mean(window_size=11).diff(n=1)
    rety = (row_h - row_h.mean()) / row_h.std()
    ind = 0
    for i in range(len(rety)):
      if rety[i] is not None and abs(rety[i]) < 0.02:
        ind = i
        break
    # print(ind)
    df = df[ind:]
    y = df['y'].to_numpy()

    # Сглаживаем данные методом Савицкого-Голая
    smoothed_y = savgol_filter(y, window_length=window_size, polyorder=polyorder)

    # Рассчитываем разницу между оригинальными и сглаженными данными
    residuals = np.abs(y - smoothed_y)

    # Фильтруем выбросы на основе порогового значения
    mask = residuals < threshold * np.max(y)
    # print(mask)
    data = {"x": [], "y": []}
    for i in range(len(mask)):
      if mask[i]:
        data['x'].append(df['x'][i])
        data['y'].append(df['y'][i])
    filtered_df = pl.from_dict(data)
    # filtered_df = pl.DataFrame({
    #     'x': df['x'][mask],
    #     'y': smoothed_y[mask]
    # })

    return filtered_df
