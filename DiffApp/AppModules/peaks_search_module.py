import numpy as np
import polars as pl
from scipy.signal import find_peaks

def find_peaks_and_centers(df: pl.DataFrame):
    # Преобразуем DataFrame в numpy массивы
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    # Находим пики
    peaks_indices, _ = find_peaks(y)
    peaks_x = x[peaks_indices]
    peaks_y = y[peaks_indices]

    # Вычисляем центры тяжести пиков
    centers_of_mass = []
    for peak_index in peaks_indices:
        # Определяем окрестность пика
        window = 115  # Размер окна, можно настроить
        start = max(0, peak_index - window)
        end = min(len(y), peak_index + window)

        # Вычисляем центр тяжести для окрестности пика
        y_sum = np.sum(y[start:end])
        if y_sum > 0:
            center_of_mass_x = np.sum(x[start:end] * y[start:end]) / y_sum
            centers_of_mass.append(center_of_mass_x)

    # Создаем DataFrame для результатов
    peaks_df = pl.DataFrame({
        'peak_x': peaks_x,
        'peak_y': peaks_y,
        'center_of_mass': centers_of_mass
    })

    return peaks_df

def find_global_peaks_and_centers(df: pl.DataFrame, window: int = 115):
    # Преобразуем DataFrame в numpy массивы
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()

    # Находим все пики
    peaks_indices, _ = find_peaks(y)

    # Создаем список для глобальных пиков
    global_peaks_x = []
    global_peaks_y = []
    global_centers_of_mass = []
    begin_ind = []
    end_ind = []
    peak_ind = []

    for peak_index in peaks_indices:
        # Определяем окрестность
        start = max(0, peak_index - window)
        end = min(len(y), peak_index + window)

        # Проверяем, является ли текущий пик глобальным максимумом
        if y[peak_index] == np.max(y[start:end]):
            global_peaks_x.append(x[peak_index])
            global_peaks_y.append(y[peak_index])
            begin_ind.append(start)
            end_ind.append(end)
            peak_ind.append(peak_index)

            # Вычисляем центр тяжести для окрестности пика
            y_sum = np.sum(y[start:end])
            if y_sum > 0:
                center_of_mass_x = np.sum(x[start:end] * y[start:end]) / y_sum
                global_centers_of_mass.append(center_of_mass_x)

    # Создаем DataFrame для результатов
    peaks_df = pl.DataFrame({
        'global_peak_x': global_peaks_x,
        'global_peak_y': global_peaks_y,
        'center_of_mass': global_centers_of_mass,
        'peak index': peak_ind,
        'region begin': begin_ind,
        'region end': end_ind
    })

    return peaks_df