import marimo

__generated_with = "0.11.13"
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
    from AppModules.calculate_module import calculate_peak_areas
    # from scipy.integrate import simpson
    from scipy.signal import find_peaks, peak_widths
    from scipy.interpolate import UnivariateSpline
    return (
        UnivariateSpline,
        alt,
        calculate_baseline,
        calculate_peak_areas,
        find_global_peaks_and_centers,
        find_peaks,
        mo,
        np,
        os,
        parse_coordinates,
        peak_widths,
        pl,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Форма ввода параметров. Параметр длины волны вводится пользователем""")
    return


@app.cell(hide_code=True)
def _(mo):
    # form = mo.ui.text_area(placeholder="Длина волны").form()
    form_base = (
        mo.md('''
        **Ввод длины волны**

        Длина волны (вещественное число)

        {wave_length}

        **Ввод гиперпараметров сглаживания**

        Размер сглаживающего окна (целое число)

        {wind_size}

        Уровень значимости (вещественное число)

        {lvl}

        Порядок полинома (целое число)

        {qut}
    ''')
        .batch(
            wave_length=mo.ui.text(),
            wind_size=mo.ui.text(),
            # label="Длина волны (вещественное число)"
            lvl=mo.ui.text(),
            qut=mo.ui.text()
        )
        .form(show_clear_button=True, bordered=False)
    )

    mo.vstack([form_base])
    return (form_base,)


@app.cell(hide_code=True)
def _(form_base):
    angle_xray = float(form_base.value['wave_length'])
    smooth_window_size = int(form_base.value['wind_size'])
    smooth_threshold_val = float(form_base.value['lvl'])
    smooth_polyorder_val = int(form_base.value['qut'])
    return (
        angle_xray,
        smooth_polyorder_val,
        smooth_threshold_val,
        smooth_window_size,
    )


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
def _():
    # smooth_window_size = mo.ui.number(start=5, stop=100, label="Размер окна для сглаживания")
    # smooth_threshold_val = mo.ui.number(start=0.001, stop=0.95, label="Уровень значимости")
    # smooth_polyorder_val = mo.ui.number(start=1, stop=25, label="Порядок полинома")
    # mo.hstack([smooth_window_size, smooth_threshold_val, smooth_polyorder_val])
    return


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
    form = (
        mo.md('''
        **Ввод гиперпараметров для выявления пиков**

        Количество значений в окрестности основного пика (целое число)

        {siz_glob}

        Количество значений в окрестности (целое число)

        {siz_peak}

        Максимальная степень полинома для построения базовой линии (целое число)

        {wert}
    ''')
        .batch(
            siz_glob=mo.ui.text(),
            siz_peak=mo.ui.text(),
            wert=mo.ui.text()
        )
        .form(show_clear_button=True, bordered=False)
    )

    mo.vstack([form])
    return (form,)


@app.cell(hide_code=True)
def _(form):
    global_peaks_window_size = int(form.value['siz_glob'])
    peaks_window_size = int(form.value['siz_peak'])
    polinom = int(form.value['wert'])
    return global_peaks_window_size, peaks_window_size, polinom


@app.cell(hide_code=True)
def _():
    # global_peaks_window_size = mo.ui.number(start=1, stop=450, label="Размер окна для выявления глобальных пиков")
    # peaks_window_size = mo.ui.number(start=1, stop=300, label="Размер окна для выявления всех пиков")
    # polinom = mo.ui.number(start=1, stop=11, label="Максимальная степень полиномиальных функций")
    # mo.hstack([global_peaks_window_size, peaks_window_size, polinom])
    return


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
                                       window_size=smooth_window_size,
                                       polyorder=smooth_polyorder_val,
                                       threshold=smooth_threshold_val)
        df_peaks_all = find_global_peaks_and_centers(df_1, window=peaks_window_size)
        df_global_peaks = find_global_peaks_and_centers(df_1,
                                                        window=global_peaks_window_size)
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
    mo.md(r"""## Базовая линия обученная на всех данных после сглаживания. Ее вывожу на графики.""")
    return


@app.cell
def _(calculate_baseline, df_1, pl, polinom):
    baseline_gr = pl.from_pandas(calculate_baseline(df_1, degree=polinom))
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
def _(
    calculate_baseline,
    df_1,
    df_global_peaks,
    peaks_window_size,
    pl,
    polinom,
):
    resulting_df = pl.DataFrame()
    leniy = []
    peaks = []

    for rowba in df_global_peaks.iter_rows(named=True):
        region_begin = rowba['region begin']
        region_end = rowba['region end']
        peaks.append(rowba['global_peak_y'])

        # Вычисляем изначальную базовую линию для среза, исключая точки в клтичестве размера окна пополам с каждой стороны
        # + int(peaks_window_size.value / 4)
        # baseline_df = baseline_gr[region_begin + int(peaks_window_size / 2):region_end - int(peaks_window_size / 2)]

        # Либо обучение на отдельной кусочной области сглаженного датафрйема
        baseline_df = pl.from_pandas(calculate_baseline(
            df_1[region_begin + int(peaks_window_size / 4):region_end - int(peaks_window_size / 4) + 1],
            degree=polinom)
        )

        # Добавляем результаты в результирующий датафрейм
        resulting_df = pl.concat([resulting_df, baseline_df])
        leniy.append(region_end - region_begin - 2*int(peaks_window_size / 4) + 1)

    resulting_df = resulting_df.with_columns(((pl.col('Baseline') + pl.col('y')) / 2).cast(pl.Float64).alias("Baseline 1/2*h"))

    igo = 0
    for und in range(len(leniy)):
        resulting_df = resulting_df.with_columns(resulting_df["Baseline 1/2*h"].scatter(range(igo, igo+leniy[und]), peaks[und]))
        # freee = pl.select(pl.repeat(peaks[und], n=leniy[und])).to_series()
        # resulting_df["Baseline 1/2*h"][igo:igo+leniy[und]] = (resulting_df["Baseline"][igo:igo+leniy[und]] + freee) / 2
        igo += leniy[und]

    resulting_df = resulting_df.with_columns(((pl.col('Baseline') + pl.col('Baseline 1/2*h')) / 2).cast(pl.Float64).alias("Baseline 1/2*h"))

    resulting_df
    return (
        baseline_df,
        igo,
        leniy,
        peaks,
        region_begin,
        region_end,
        resulting_df,
        rowba,
        und,
    )


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
            degree=polinom)
        )

        # Добавляем результаты в результирующий датафрейм
        res_df = pl.concat([res_df, base_df])
    res_df
    return base_df, reg_begin, reg_end, res_df, rowbab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Просмотр площадей пиков/инетгральной интенсивности каждого из пиков""")
    return


@app.cell
def _(baseline_gr, calculate_peak_areas, df_global_peaks, peaks_window_size):
    integ_intensity = calculate_peak_areas(baseline_gr, df_global_peaks, wind_size=int(peaks_window_size / 2))
    integ_intensity
    return (integ_intensity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ввод параметров ширины и высоты графика, просмотр графиков""")
    return


@app.cell(hide_code=True)
def _(mo):
    form_gr = (
        mo.md('''
        **Ввод масштаба графика**

        Задать ширину графика (целое число)

        {gr_width}

        Задать высоту графика (целое число)

        {gr_h}
    ''')
        .batch(
            gr_width=mo.ui.text(),
            gr_h=mo.ui.text()
        )
        .form(show_clear_button=True, bordered=False)
    )

    mo.vstack([form_gr])
    return (form_gr,)


@app.cell(hide_code=True)
def _(form_gr):
    graph_width = int(form_gr.value['gr_width'])
    graph_height = int(form_gr.value['gr_h'])
    return graph_height, graph_width


@app.cell(hide_code=True)
def _():
    # graph_height = mo.ui.number(start=100, stop=3000, label="Значение высоты графика")
    # graph_width = mo.ui.number(start=100, stop=3000, label="Значение ширины графика")
    # mo.hstack([graph_width, graph_height])
    return


@app.cell
def _(alt, df, graph_height, graph_width):
    if df is not None:
        lines111 = (
            alt.Chart(df.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity')).properties(
                width=graph_width,
                height=graph_height
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
                width=graph_width,
                height=graph_height
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
def _(alt, baseline_gr, df, df_1, df_global_peaks, graph_height, graph_width):
    if df is not None:
        lines_show = (
            alt.Chart(df_1.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity'), color=alt.value('blue')).properties(
                width=graph_width,
                height=graph_height
            )
        )

        # Данные для пунктирных линий и меток
        baseline_chart = (
            alt.Chart(baseline_gr)
            .mark_line(color='green').encode(
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
def _(
    alt,
    df,
    df_1,
    df_global_peaks,
    graph_height,
    graph_width,
    ind,
    leniy,
    resulting_df,
):
    if df is not None:
        lines_showff = (
            alt.Chart(df_1.to_pandas())
            .mark_line()
            .encode(x=alt.X("x", title='Angle 2θ (°)'), y=alt.Y("y", title='Intensity'), color=alt.value('blue')).properties(
                width=graph_width,
                height=graph_height
            )
        )

        baseline_char = []
        igpp = 0

        # Iterate over the baseline columns and create a chart for each
        for loki in range(len(leniy)):
            chartii = (
                alt.Chart(resulting_df[igpp:igpp + leniy[loki]])
                .mark_line()
                .encode(
                    x=alt.X('x', title=''),
                    y=alt.Y('Baseline', title=''),
                    color=alt.value('green')  # Different colors for each line
                    )
                )
            baseline_char.append(chartii)
            igpp += leniy[ind]

        base_chartiii = alt.layer(*baseline_char)

        # Данные для пунктирных линий и меток из DataFrame с базовой линией
        # baseline_chartff = (
        #     alt.Chart(resulting_df)
        #     .mark_line(color='green').encode(
        #         x=alt.X('x', title=''),
        #         y=alt.Y('Baseline', title='')
        #     )
        # )

        # Пунктирные линии
        dotted_lines_showff = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_rule(strokeWidth=1.5, color='red')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        # Метки
        text_labels_showff = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

    char_final = lines_showff + base_chartiii + dotted_lines_showff + text_labels_showff
    char_final
    return (
        base_chartiii,
        baseline_char,
        char_final,
        chartii,
        dotted_lines_showff,
        igpp,
        lines_showff,
        loki,
        text_labels_showff,
    )


@app.cell
def _(
    alt,
    df,
    df_1,
    df_global_peaks,
    graph_height,
    graph_width,
    leniy,
    resulting_df,
):
    # Assuming df is your DataFrame and it has columns "x", "y", "Baseline", "Baseline 1/2*h"
    if df is not None:
        # Create the initial line chart for the "y" column
        lines_show_ff = (
            alt.Chart(df_1.to_pandas())
            .mark_line()
            .encode(
                x=alt.X("x", title='Angle 2θ (°)'),
                y=alt.Y("y", title='Intensity'),
                color=alt.value('blue')
            )
            .properties(
                width=graph_width,
                height=graph_height
            )
        )

        # List of columns to plot as separate lines
        baseline_columns = ["Baseline", "Baseline 1/2*h"]

        # Create a list to store the charts
        baseline_charts = []
        igp = 0

        # Iterate over the baseline columns and create a chart for each
        for ind in range(len(leniy)):
            for coli in baseline_columns:
                chart = (
                    alt.Chart(resulting_df[igp:igp + leniy[ind]])
                    .mark_line()
                    .encode(
                        x=alt.X('x', title=''),
                        y=alt.Y(coli, title=''),
                        color=alt.value('green' if coli == "Baseline" else 'orange')  # Different colors for each line
                    )
                )
                baseline_charts.append(chart)
            igp += leniy[ind]

        # Combine all baseline charts into a single chart
        baseline_chart_ff = alt.layer(*baseline_charts)

        # Пунктирные линии
        dotted_lines_show_ff = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_rule(strokeWidth=1.5, color='red')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        # Метки
        text_labels_show_ff = (
            alt.Chart(df_global_peaks.to_pandas())
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

    # Combine all charts together
    final_chart = lines_show_ff + baseline_chart_ff + dotted_lines_show_ff + text_labels_show_ff

    # Display the final chart
    final_chart
    return (
        baseline_chart_ff,
        baseline_charts,
        baseline_columns,
        chart,
        coli,
        dotted_lines_show_ff,
        final_chart,
        igp,
        ind,
        lines_show_ff,
        text_labels_show_ff,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Вычисление максимальных значений интегральной и максимальной интенсивностей для вычисления относительных значений интенсивностей в процентах""")
    return


@app.cell(hide_code=True)
def _(baseline_gr, df_1, df_global_peaks, pl):
    dating_max_intensity = -1

    for indx, rowi in enumerate(df_global_peaks.iter_rows(), start=1):
        calcing = df_1.row(by_predicate=(pl.col("x") == rowi[0]))[1] - baseline_gr.row(by_predicate=(pl.col("x") == rowi[0]))[2]
        if calcing > dating_max_intensity:
            dating_max_intensity = calcing

    # print(dating_max_intensity)
    return calcing, dating_max_intensity, indx, rowi


@app.cell(hide_code=True)
def _(integ_intensity):
    dating_max_integral_intensity = -1

    for indeex, rowib in enumerate(integ_intensity.iter_rows(), start=1):
        calcin = rowib[2]
        if calcin > dating_max_integral_intensity:
            dating_max_integral_intensity = calcin

    # print(dating_max_integral_intensity)
    return calcin, dating_max_integral_intensity, indeex, rowib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Вывод таблицы характеристик каждого из пиков.""")
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
        distance = angle_xray / (2 * np.sin(np.radians(angle) / 2))  # Расчет расстояния между плоскостями
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
            "Интегральная интенсивность (имп^2 / сек^2)": integral_intensity,
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
