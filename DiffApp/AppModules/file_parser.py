import polars as pl
from AppModules.smoothing_module import smooth_and_filter_peaks

def parse_coordinates(file_path):
    # Создаем пустой список для хранения координат
    coordinates = []
    
    # Открываем файл и считываем его построчно
    with open(file_path, 'r') as file:
        for line in file:
            # Убираем лишние пробелы и символы новой строки
            line = line.strip()
            # Проверяем, что строка не пустая
            if line:
                # Исключение делаем для считывания специфических файлов с дифрактограммами воды
                try:
                    # Разделяем строку на x и y
                    x, y = map(float, line.split())
                    # Добавляем координаты в список
                    coordinates.append((x, y))
                except ValueError:
                    continue
                
    
    # Создаем DataFrame из списка координат
    df = pl.DataFrame(coordinates, schema=['x', 'y'], orient="row")
    return df
