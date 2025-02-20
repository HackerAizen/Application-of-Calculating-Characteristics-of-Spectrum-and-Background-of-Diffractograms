import marimo

__generated_with = "0.11.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Для работы нужно подключить библиотеки marimo, numpy, polars, os, sys, altair, а также sklearn и scipy для вычислений.""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import os
    import sys
    import altair as alt
    from AppModules.file_parser import parse_coordinates
    from AppModules.peaks_search_module import find_global_peaks_and_centers
    from AppModules.smoothing_module import smooth_and_filter_peaks
    from AppModules.calculate_module import calculate_baseline
    from scipy.integrate import simpson
    from scipy.signal import find_peaks, peak_widths
    from scipy.interpolate import UnivariateSpline
    return (
        UnivariateSpline,
        alt,
        calculate_baseline,
        find_global_peaks_and_centers,
        find_peaks,
        mo,
        np,
        os,
        parse_coordinates,
        peak_widths,
        pl,
        simpson,
        smooth_and_filter_peaks,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Загрузка и выбор файла производится, одновременно, следует выбирать один файл.""")
    return


@app.cell(hide_code=True)
def _(mo):
    # f = mo.ui.file(['.txt'], False)
    f = mo.ui.file_browser()
    f
    return (f,)


@app.cell
def _(mo):
    mo.md(r"""## Окна ввода параметров. Параметр длины волны вводится пользователем (ГМ сказала, что надо так)""")
    return


@app.cell(hide_code=True)
def _(mo):
    angle_xray = mo.ui.number(start=0.001, stop=1300, label="Длина волны")
    mo.hstack([angle_xray])
    return (angle_xray,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Подбор гиперпараметров для сглаживания кривой. Вводится пользователем (для coord.txt оптимально брать размер окна - 12, уровень значимости - 0.05, порядок полинома - 3).
        ## Для coord_1.txt пытался взять 20, 0.003, 3 соответственно.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    smooth_window_size = mo.ui.number(start=5, stop=100, label="Размер окна для сглаживания")
    smooth_threshold_val = mo.ui.number(start=0.001, stop=0.95, label="Уровень значимости")
    smooth_polyorder_val = mo.ui.number(start=1, stop=25, label="Порядок полинома")
    mo.hstack([smooth_window_size, smooth_threshold_val, smooth_polyorder_val])
    return smooth_polyorder_val, smooth_threshold_val, smooth_window_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Подбор гиперпараметров для выявления пиков. Вводится пользователем (для coord.txt брал глобальные пики окно - 115, все пики - 80, степень полинома - 1).
        ## Для coord_1.txt пытался взять 155, 110, 2 соответственно.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    global_peaks_window_size = mo.ui.number(start=1, stop=450, label="Размер окна для выявления глобальных пиков")
    peaks_window_size = mo.ui.number(start=1, stop=300, label="Размер окна для выявления всех пиков")
    polinom = mo.ui.number(start=1, stop=11, label="Максимальная степень полиномиальных функций")
    mo.hstack([global_peaks_window_size, peaks_window_size, polinom])
    return global_peaks_window_size, peaks_window_size, polinom


@app.cell
def _(
    f,
    find_global_peaks_and_centers,
    global_peaks_window_size,
    parse_coordinates,
    peaks_window_size,
    smooth_and_filter_peaks,
    smooth_polyorder_val,
    smooth_threshold_val,
    smooth_window_size,
):
    # df = pl.read_csv(f.value[0].contents, schema = {'x': pl.Float64, 'y': pl.Float64}, has_header=False, separator='\t')
    df = None
    df_1 = None
    df_global_peaks = None
    if f.value != ():
        # df = parse_coordinates(os.path.abspath("DataTXT/" + f.value[0].name))
        df = parse_coordinates(f.value[0].path)
        df_1 = smooth_and_filter_peaks(df,
                                       window_size=smooth_window_size.value,
                                       polyorder=smooth_polyorder_val.value,
                                       threshold=smooth_threshold_val.value)
        df_peaks_all = find_global_peaks_and_centers(df_1, window=peaks_window_size.value)
        df_global_peaks = find_global_peaks_and_centers(df_1,
                                                        window=global_peaks_window_size.value)
    # df_1
    return df, df_1, df_global_peaks, df_peaks_all


@app.cell
def _():
    # df_peaks_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Вывод найденных глобальных пиков с координатами значения угла и интенсивности, рассчитанным центром масс, порядковым индексом пика в polars dataframe, началом и концом промежутка области пика с учетом окна введенного. Далее использовалась идея поиска интегральной интенсивности и расчета базовой линии с помощью взятия среза polars dataframe по промежуткам и дальнейшей работы с ними.""")
    return


@app.cell
def _(df_global_peaks):
    df_global_peaks
    return


@app.cell
def _(df_global_peaks, df_peaks_all):
    df_cleaned = df_peaks_all.join(df_global_peaks, on="global_peak_x", how="anti")
    # df_cleaned
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Пытался сделать поиск области пика через peak_widths, но безуспешно, к сожалению. Поэтому сделал функцию ниже и распишу позже.""")
    return


@app.cell
def _():
    # xxx = df_1['x'].to_numpy().copy()
    # yyy = df_1['y'].to_numpy().copy()
    # peaks, _ = find_peaks(yyy)
    # i = 0 
    # results_half = peak_widths(xxx, peaks, rel_height=0.5)
    # results_full = peak_widths(xxx, peaks, rel_height=1)
    # print(peaks)
    # for boo in peaks:
    #     st = max(0, boo - 115)
    #     en = min(len(yyy), boo + 115)

    #     if yyy[boo] == np.max(yyy[st:en]):
    #         print(st, en)
    #         results_half = peak_widths(xxx[st:en].copy(), np.array([int(xxx[boo])]).copy(), rel_height=0.5)
    #         results_full = peak_widths(xxx[st:en].copy(), np.array([int(xxx[boo])]).copy(), rel_height=1)
    #         print(boo)
    #         print(xxx[boo])
    #         print(results_half)
    #         print(results_full)
            # results_half = peak_widths(xxx, peaks, rel_height=0.5)
            # results_full = peak_widths(xxx, peaks, rel_height=1)
            # print(results_half[0][boo])
            # print(results_full[0][boo])

    #results_half = peak_widths(df_1['x'].to_numpy(), df_global_peaks['peak index'].to_numpy(), rel_height=0.5)
    #results_half[0]
    #results_half[1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Базовая линия обученная на всех данных после сглаживания. Ее вывожу на графики.""")
    return


@app.cell
def _(calculate_baseline, df_1, pl, polinom):
    baseline_gr = pl.from_pandas(calculate_baseline(df_1, degree=polinom.value))
    baseline_gr
    return (baseline_gr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Базовая линия для каждого пика в отдельности. Пробовал два варианта:
        - Брал срез изначальной базовой линии по промежуткам c учетом окна.
        - Обучал кусочно линейную регрессию из calculate_module по областям промежуткам с учетом окна.
        """
    )
    return


@app.cell
def _(baseline_gr, df_global_peaks, peaks_window_size, pl):
    resulting_df = pl.DataFrame()

    for rowba in df_global_peaks.iter_rows(named=True):
        region_begin = rowba['region begin']
        region_end = rowba['region end']

        # Вычисляем изначальную базовую линию для среза, исключая точки в клтичестве размера окна пополам с каждой стороны
        # + int(peaks_window_size.value / 4)
        baseline_df = baseline_gr[region_begin + int(peaks_window_size.value / 2):region_end - int(peaks_window_size.value / 2)]

        # Либо обучение на отдельной кусочной области сглаженного датафрйема
        # baseline_df = pl.from_pandas(calculate_baseline(
        #     df_1[region_begin:region_end],
        #     degree=polinom.value)
        # )

        # Добавляем результаты в результирующий датафрейм
        resulting_df = pl.concat([resulting_df, baseline_df])

    resulting_df = resulting_df.with_columns((pl.col('Baseline') * 2).cast(pl.Float64).alias("Baseline 1/2*h"))
    resulting_df
    return baseline_df, region_begin, region_end, resulting_df, rowba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Базовая линия для пиков, которые не вошли в глобальные но могут быть важны""")
    return


@app.cell
def _(calculate_baseline, df_1, df_cleaned, pl, polinom):
    res_df = pl.DataFrame()
    for rowbab in df_cleaned.iter_rows(named=True):
        reg_begin = rowbab['region begin']
        reg_end = rowbab['region end']

        # Вычисляем базовую линию для среза, исключая n размер окна на 4 точек с каждой стороны.
        #  - int(peaks_window_size.value / 4)
        base_df = pl.from_pandas(calculate_baseline(
            df_1[reg_begin:reg_end + 1],
            degree=polinom.value)
        )

        # Добавляем результаты в результирующий датафрейм
        res_df = pl.concat([res_df, base_df])
    res_df
    return base_df, reg_begin, reg_end, res_df, rowbab


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## А вот отсюда начинаются все проблемы. Мое предположение - некорректное вычисление базовой линии и работа с индексами и срезами.

        Идея в том, чтобы:

        - пройтись по датафрейму с глобальными пиками таким образом 
        - взять значения начала и конца регионов из датафрейма с пиками
        - на нем взять срез с добавлением значения окна либо уже обученной базовой линии на всем массиве данных, либо взять часть данных срезом и обучить только на них линейную регрессию для базовой линии
        -  добавляем в результирующий датафрейм базовую линию
        -  далее проходимся снова циклом по массиву глобальных пиков и формируем небольшой массив данных с значениями границ базовой линии для дальнешего вычитания ее значений из y данных для ввода в функции simpson или trapz для расчета площади пика
        -  в переменные суем numpy массивы с значениями столбцов координат дифрактограммы (2 тета и интенсивности), а также значения координат по оси y (интенсивностей)
        -  дальше вычитаем значения базовой линии из интенсивности y
        -  закомментированы некоторые эвристики для взятия только тех значений которые выше базовой линии (иначе площади уходят в сильный минус слишком)
        -  закомментировал:
           *  h2_x_values = x_values[cory >=0],
           * x_values = x_values[corrected_y >=0],
           *  cory = cory[cory >= 0],
           *  corrected_y = corrected_y[corrected_y >=0]
        -  дальше вычисляем методом simpson или trapz площадь на промежутке x_values и вычисленном скорректированном y после вычитания базовой линии, также пытался вычислить на половине высоты, но, к сожалению, там тоже не работает
        -  добавляем в массив значения начала и конца промежутка, вычисленную площадь пика и неправильную вычисленную площадь пика на половине высоты
        -  формируем датафрейм с результатами
        """
    )
    return


@app.cell
def _(pl, simpson):
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
            print(len(corrected_y), len(x_values))

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
            print(area_h)

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
    return (calculate_peak_areas,)


@app.cell
def _(mo):
    mo.md(r"""## Просмотр площадей пиков/инетгральной интенсивности каждого из пиков""")
    return


@app.cell
def _(
    baseline_gr,
    calculate_peak_areas,
    df_global_peaks,
    peaks_window_size,
):
    integ_intensity = calculate_peak_areas(baseline_gr, df_global_peaks, wind_size=int(peaks_window_size.value / 2))
    integ_intensity
    return (integ_intensity,)


@app.cell
def _(mo):
    mo.md(r"""## Ввод параметров ширины и высоты графика, просмотр графиков""")
    return


@app.cell(hide_code=True)
def _(mo):
    graph_height = mo.ui.number(start=100, stop=3000, label="Значение высоты графика")
    graph_width = mo.ui.number(start=100, stop=3000, label="Значение ширины графика")
    mo.hstack([graph_width, graph_height])
    return graph_height, graph_width


@app.cell
def _(alt, df, graph_height, graph_width):
    if df is not None:
        lines111 = (
            alt.Chart(df.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity')).properties(
                width=graph_width.value,
                height=graph_height.value
            )
        )
    (lines111).properties(title=alt.Title(text='Дифрактограмма до обработки'))
    return (lines111,)


@app.cell(hide_code=True)
def _(UnivariateSpline, df_1, pl):
    splinen = UnivariateSpline(df_1['x'].to_numpy(), df_1['y'].to_numpy(), k=5)
    baseline115 = splinen(df_1['x'].to_numpy())
    df118 = df_1.with_columns(pl.Series('Baseline', baseline115))
    return baseline115, df118, splinen


@app.cell
def _(
    alt,
    baseline_gr,
    df,
    df_1,
    df_cleaned,
    df_global_peaks,
    graph_height,
    graph_width,
):
    if df is not None:
        # lines = (
        #     alt.Chart(df.to_pandas())
        #     .mark_line()
        #     .encode(x="x", y="y").properties(
        #         width=graph_width.value,
        #         height=graph_height.value
        #     )
        # )

        lines_1 = (
            alt.Chart(df_1.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity')).properties(
                width=graph_width.value,
                height=graph_height.value
            )
        )

        # Данные для пунктирных линий и меток
        df_global_peaks_pd = df_global_peaks.to_pandas()  # Преобразуем в Pandas DataFrame

    # Пунктирные линии
        dotted_lines = (
            alt.Chart(df_global_peaks_pd)
            .mark_rule(strokeWidth=2, color='purple')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        dotted_lines_1 = (
            alt.Chart(df_cleaned.to_pandas())
            .mark_rule(strokeDash=[15, 15], color='green')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        baseline_chart_show = (
            alt.Chart(baseline_gr)
            .mark_line(color='black').encode(
                x=alt.X('x', title=''),
                y=alt.Y('Baseline', title='')
            )
        )

    # Метки
        text_labels = (
            alt.Chart(df_global_peaks_pd)
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

        text_labels_1 = (
            alt.Chart(df_cleaned.to_pandas())
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

    (lines_1 + dotted_lines + dotted_lines_1 + text_labels + text_labels_1 + baseline_chart_show).properties(title=alt.Title(text='Дифрактограмма после обработки с помеченными пиками, их центрами тяжести и ее базовой линией'))
    return (
        baseline_chart_show,
        df_global_peaks_pd,
        dotted_lines,
        dotted_lines_1,
        lines_1,
        text_labels,
        text_labels_1,
    )


@app.cell
def _(
    alt,
    baseline_gr,
    df,
    df_1,
    df_global_peaks,
    graph_height,
    graph_width,
):
    if df is not None:
        lines_show = (
            alt.Chart(df_1.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity'), color=alt.value('black')).properties(
                width=graph_width.value,
                height=graph_height.value
            )
        )

        # Данные для пунктирных линий и меток
        baseline_chart = (
            alt.Chart(baseline_gr)
            .mark_line(color='orange').encode(
                x=alt.X('x', title=''),
                y=alt.Y('Baseline', title='')
            )
        )

        # Пунктирные линии
        dotted_lines_show = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_rule(strokeWidth=1.5, color='red')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        # Метки
        text_labels_show = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

    lines_show + baseline_chart + dotted_lines_show + text_labels_show
    return baseline_chart, dotted_lines_show, lines_show, text_labels_show


@app.cell
def _(mo):
    mo.md(r"""## Вычисление максимальных значений интегральной и максимальной интенсивностей для вычисления относительных значений интенсивностей в процентах""")
    return


@app.cell
def _(baseline_gr, df_1, df_global_peaks, pl):
    dating_max_intensity = -1

    for indx, rowi in enumerate(df_global_peaks.iter_rows(), start=1):
        calcing = df_1.row(by_predicate=(pl.col("x") == rowi[0]))[1] - baseline_gr.row(by_predicate=(pl.col("x") == rowi[0]))[2]
        if calcing > dating_max_intensity:
            dating_max_intensity = calcing

    # print(dating_max_intensity)
    return calcing, dating_max_intensity, indx, rowi


@app.cell
def _(integ_intensity):
    dating_max_integral_intensity = -1

    for indeex, rowib in enumerate(integ_intensity.iter_rows(), start=1):
        calcin = rowib[2]
        if calcin > dating_max_integral_intensity:
            dating_max_integral_intensity = calcin

    # print(dating_max_integral_intensity)
    return calcin, dating_max_integral_intensity, indeex, rowib


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Вывод таблицы характеристик каждого из пиков, как просит ГМ. Главная проблема заключается в расчете интегральной интенсивности каждого пика на данных coord_1, отрицательная площадь пиков и отношение бета, которая оказывается больше 1 или отрицательным из-за некорректного вычисления площади пиков.

        Также не понимаю как вычислять площадь пика на высоте 1/2*h и уровень доверия R.
        """
    )
    return


@app.cell(hide_code=True)
def _(
    angle_xray,
    baseline_gr,
    dating_max_integral_intensity,
    dating_max_intensity,
    df_1,
    df_global_peaks,
    integ_intensity,
    mo,
    np,
    pl,
):
    # data = [
    #     {"Номер пика N": 100, "Угол 2θ (°)": 30, "Расстояние между плоскостями решетки": 23, "Интегральная интенсивность (имп / сек)": 25, "Максимальная интенсивность (имп / сек)": 45, "Относительная интегральная интенсивность (%)": 43.3, "Относительная максимальная интенсивность (%)": 67.8, "Уровень доверия R (площадь пика, описывающегося функцией f / площадь пика)": 0.65, "Площадь пика на уровне 1/2*h": 1321.2, "Отношение β (интегральная интенсивность пика / максимальная интенсивность пика)": 0.51}
    # ]

    data = []

    for idx, row in enumerate(df_global_peaks.iter_rows(), start=1):
        angle = row[0]  # Угол (в градусах)
        distance = angle_xray.value / (2 * np.sin(np.radians(angle) / 2))  # Расчет расстояния между плоскостями
        max_intensity = df_1.row(by_predicate=(pl.col("x") == row[0]))[1] - baseline_gr.row(by_predicate=(pl.col("x") == row[0]))[2] # - resulting_df.row(by_predicate=(pl.col("x") == row[0]))[2] # - baseline_gr.row(by_predicate=(pl.col("x") == row[0]))[2]
        # print(df_1.row(by_predicate=(pl.col("x") == row[0]))[1])
        # print(resulting_df.row(by_predicate=(pl.col("x") == row[0]))[2])
        ratio_max_intensity = max_intensity / dating_max_intensity
        integral_intensity = integ_intensity[idx - 1]['peak_area'][0]
        # print(integ_intensity[idx - 1]['peak_area'][0])
        ratio_integral_intensity = integral_intensity / dating_max_integral_intensity
        integral_intensity_h2 = integ_intensity[idx - 1]['peak_area_h2'][0]
        data.append({
            "Номер пика N": idx,
            "Угол 2θ (°)": angle,
            "Расстояние между плоскостями решетки": distance,
            "Максимальная интенсивность (имп / сек)": max_intensity,
            "Относительная максимальная интенсивность (%)": ratio_max_intensity * 100,
            "Интегральная интенсивность (имп / сек)": integral_intensity,
            "Относительная интегральная интенсивность (%)": ratio_integral_intensity * 100,
            "Отношение β (интегральная интенсивность пика / максимальная интенсивность пика)": integral_intensity / max_intensity,
            "Площадь пика на уровне 1/2*h": integral_intensity_h2
            # "Уровень доверия R (площадь пика, описывающегося функцией f / площадь пика)": -1
        })

    # columns = [ {"label": "Angle (°)", "key": "angle"}, {"label": "Diffraction Integral Intensity", "key": "diffraction_integral_intensity"}, {"label": "Diffraction Maximal Intensity", "key": "diffraction_maximal_intensity"}, {"label": "Ratio (Integral / Max)", "key": "ratio"},]

    table = mo.ui.table(data=data)

    mo.vstack([table])
    return (
        angle,
        data,
        distance,
        idx,
        integral_intensity,
        integral_intensity_h2,
        max_intensity,
        ratio_integral_intensity,
        ratio_max_intensity,
        row,
        table,
    )


if __name__ == "__main__":
    app.run()
