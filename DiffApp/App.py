import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Для работы нужно подключить библиотеки marimo, numpy, polars, os, sys, altair, а также sklearn и scipy для вычислений.""")
    return


@app.cell(hide_code=True)
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
    from scipy.signal import find_peaks, peak_widths
    import json
    return (
        alt,
        calculate_baseline,
        calculate_peak_areas,
        find_global_peaks_and_centers,
        find_peaks,
        json,
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
    mo.md(r"""## Форма ввода параметров. Параметр длины волны вводится пользователем. Гиперпараметры сглаживания также выбираются пользователем""")
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
def _(form_base, mo):
    try:
        angle_xray = float(form_base.value['wave_length'])
        smooth_window_size = int(form_base.value['wind_size'])
        smooth_threshold_val = float(form_base.value['lvl'])
        smooth_polyorder_val = int(form_base.value['qut'])
    except TypeError:
        with mo.redirect_stdout():
            print("Сначала введите числовые значения в форму параметров выше!")
    except ValueError:
        with mo.redirect_stdout():
            print("Числа с плавающей точкой записываются в форму через точку, а не через запятую!")
    return (
        angle_xray,
        smooth_polyorder_val,
        smooth_threshold_val,
        smooth_window_size,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Ввод гиперпараметров для выявления пиков.""")
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

        Сколько k значимых глобальных пиков учитывать?

        {quantity_peaks}
    ''')
        .batch(
            siz_glob=mo.ui.text(),
            siz_peak=mo.ui.text(),
            wert=mo.ui.text(),
            quantity_peaks=mo.ui.text()
        )
        .form(show_clear_button=True, bordered=False)
    )

    mo.vstack([form])
    return (form,)


@app.cell(hide_code=True)
def _(form, mo):
    try:
        global_peaks_window_size = int(form.value['siz_glob'])
        peaks_window_size = int(form.value['siz_peak'])
        polinom = int(form.value['wert'])
        qunatity_peak_kol = int(form.value['quantity_peaks'])
    except TypeError:
        with mo.redirect_stdout():
            print("Сначала введите числовые значения в форму гиперпараметров выше с корректными типами данных!")
    return (
        global_peaks_window_size,
        peaks_window_size,
        polinom,
        qunatity_peak_kol,
    )


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Вывод найденных глобальных пиков с координатами значения угла и интенсивности, рассчитанным центром масс, порядковым индексом пика в polars dataframe, началом и концом промежутка области пика с учетом окна введенного. Далее использовалась идея поиска интегральной интенсивности и расчета базовой линии с помощью взятия среза polars dataframe по промежуткам и дальнейшей работы с ними.""")
    return


@app.cell(hide_code=True)
def _(df_global_peaks):
    df_global_peaks
    return


@app.cell(hide_code=True)
def _(df_global_peaks, json, mo, pl):
    tablet_global_peaks = pl.DataFrame(df_global_peaks).to_pandas() # .to_csv(index=False)

    download_tablet_global_peaks_csv = mo.download(
        data=tablet_global_peaks.to_csv(index=False).encode("utf-8"),
        filename="info_global_peaks_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с информацией о глобальных пиках в CSV",
    )

    download_tablet_global_peaks_json = mo.download(
        data=json.dumps(tablet_global_peaks.to_json()).encode("utf-8"),
        filename="info_global_peaks_table.json",
        mimetype="application/json",
        label="Скачать таблицу с информацией о глобальных пиках в JSON",
    )

    mo.hstack([download_tablet_global_peaks_csv, download_tablet_global_peaks_json])
    return (
        download_tablet_global_peaks_csv,
        download_tablet_global_peaks_json,
        tablet_global_peaks,
    )


@app.cell(hide_code=True)
def _(df_global_peaks, df_peaks_all):
    df_cleaned = df_peaks_all.join(df_global_peaks, on="global_peak_x", how="anti")
    # df_cleaned
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(df_global_peaks, qunatity_peak_kol):
    df_sel_peaks = df_global_peaks.top_k(qunatity_peak_kol, by="global_peak_y")
    df_sel_peaks
    return (df_sel_peaks,)


@app.cell(hide_code=True)
def _(df_sel_peaks, json, mo, pl):
    tablet_sel_peaks = pl.DataFrame(df_sel_peaks).to_pandas() # .to_csv(index=False)

    download_tablet_sel_peaks_csv = mo.download(
        data=tablet_sel_peaks.to_csv(index=False).encode("utf-8"),
        filename="info_k_significant_peaks_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с информацией о k значимых пиках в CSV",
    )

    download_tablet_sel_peaks_json = mo.download(
        data=json.dumps(tablet_sel_peaks.to_json()).encode("utf-8"),
        filename="info_k_significant_peaks_table.json",
        mimetype="application/json",
        label="Скачать таблицу с информацией о k значимых пиках в JSON",
    )

    mo.hstack([download_tablet_sel_peaks_csv, download_tablet_sel_peaks_json])
    return (
        download_tablet_sel_peaks_csv,
        download_tablet_sel_peaks_json,
        tablet_sel_peaks,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Базовая линия обученная на всех данных после сглаживания.""")
    return


@app.cell(hide_code=True)
def _(calculate_baseline, df_1, pl, polinom):
    baseline_gr = pl.from_pandas(calculate_baseline(df_1, degree=polinom))
    baseline_gr
    return (baseline_gr,)


@app.cell(hide_code=True)
def _(baseline_gr, json, mo, pl):
    tablet_baseline_gr = pl.DataFrame(baseline_gr).to_pandas() # .to_csv(index=False)

    download_tablet_baseline_gr_csv = mo.download(
        data=tablet_baseline_gr.to_csv(index=False).encode("utf-8"),
        filename="info_baseline_gr_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с информацией о базовой линии дифрактограммы в CSV",
    )

    download_tablet_baseline_gr_json = mo.download(
        data=json.dumps(tablet_baseline_gr.to_json()).encode("utf-8"),
        filename="info_baseline_gr_table.json",
        mimetype="application/json",
        label="Скачать таблицу с информацией о базовой линии дифрактограммы в JSON",
    )

    mo.hstack([download_tablet_baseline_gr_csv, download_tablet_baseline_gr_json])
    return (
        download_tablet_baseline_gr_csv,
        download_tablet_baseline_gr_json,
        tablet_baseline_gr,
    )


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


@app.cell(hide_code=True)
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
def _(json, mo, pl, resulting_df):
    tablet_baseline_dop = pl.DataFrame(resulting_df).to_pandas() # .to_csv(index=False)

    download_tablet_resulting_df_csv = mo.download(
        data=tablet_baseline_dop.to_csv(index=False).encode("utf-8"),
        filename="info_baseline_sep_peak_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с информацией о базовой линии каждого пика в CSV",
    )

    download_tablet_resulting_df_json = mo.download(
        data=json.dumps(tablet_baseline_dop.to_json()).encode("utf-8"),
        filename="info_baseline_sep_peak_table.json",
        mimetype="application/json",
        label="Скачать таблицу с информацией о базовой линии каждого пика в JSON",
    )

    mo.hstack([download_tablet_resulting_df_csv, download_tablet_resulting_df_json])
    return (
        download_tablet_resulting_df_csv,
        download_tablet_resulting_df_json,
        tablet_baseline_dop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Базовая линия для пиков, которые не вошли в глобальные но могут быть важны""")
    return


@app.cell(hide_code=True)
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
def _(json, mo, pl, res_df):
    tablet_baseline_wert = pl.DataFrame(res_df).to_pandas() # .to_csv(index=False)

    download_tablet_res_df_csv = mo.download(
        data=tablet_baseline_wert.to_csv(index=False).encode("utf-8"),
        filename="info_baseline_sep_peak_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с информацией о базовой линии побочных пиков в CSV",
    )

    download_tablet_res_df_json = mo.download(
        data=json.dumps(tablet_baseline_wert.to_json()).encode("utf-8"),
        filename="info_baseline_sep_peak_table.json",
        mimetype="application/json",
        label="Скачать таблицу с информацией о базовой линии побочных пиков в JSON",
    )

    mo.hstack([download_tablet_res_df_csv, download_tablet_res_df_json])
    return (
        download_tablet_res_df_csv,
        download_tablet_res_df_json,
        tablet_baseline_wert,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Просмотр площадей пиков/интегральной интенсивности каждого из пиков""")
    return


@app.cell(hide_code=True)
def _(baseline_gr, calculate_peak_areas, df_global_peaks, peaks_window_size):
    integ_intensity = calculate_peak_areas(baseline_gr, df_global_peaks, wind_size=int(peaks_window_size / 2))
    integ_intensity
    return (integ_intensity,)


@app.cell(hide_code=True)
def _(integ_intensity, json, mo, pl):
    tablet_integ = pl.DataFrame(integ_intensity).to_pandas() # .to_csv(index=False)

    download_tablet_integ_csv = mo.download(
        data=tablet_integ.to_csv(index=False).encode("utf-8"),
        filename="integ_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу с интегральными интенсивностями пиков в CSV",
    )

    download_tablet_integ_json = mo.download(
        data=json.dumps(tablet_integ.to_json()).encode("utf-8"),
        filename="integ_table.json",
        mimetype="application/json",
        label="Скачать таблицу с интегральными интенсивностями пиков в JSON",
    )

    mo.hstack([download_tablet_integ_csv, download_tablet_integ_json])
    return download_tablet_integ_csv, download_tablet_integ_json, tablet_integ


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
def _(form_gr, mo):
    try:
        graph_width = int(form_gr.value['gr_width'])
        graph_height = int(form_gr.value['gr_h'])
    except TypeError:
        with mo.redirect_stdout():
            print("Сначала введите числовые значения в форму масштаба графика выше!")
    return graph_height, graph_width


@app.cell(hide_code=True)
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
    tutu = (lines111).properties(title=alt.Title(text='Дифрактограмма до обработки'))
    tutu
    return lines111, tutu


@app.cell(hide_code=True)
def _(mo, tutu):
    tutu.save('original_diffractogram.png')
    download_tutu = mo.download(data="original_diffractogram.png", filename="original_diffractogram.png", mimetype="image/png", label="Скачать дифрактограмму до обработки в формате PNG")

    tutu.save('original_diffractogram.svg')
    download_tutu_svg = mo.download(data="original_diffractogram.svg", filename="original_diffractogram.svg", mimetype="image/svg", label="Скачать дифрактограмму до обработки в формате SVG")

    mo.hstack([download_tutu, download_tutu_svg])
    return download_tutu, download_tutu_svg


@app.cell(hide_code=True)
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

    chart_pupu = (lines_1 + dotted_lines + dotted_lines_1 + text_labels + text_labels_1 + baseline_chart_show).properties(title=alt.Title(text='Дифрактограмма после обработки с помеченными пиками, их центрами тяжести и ее базовой линией'))
    chart_pupu
    return (
        baseline_chart_show,
        chart_pupu,
        df_global_peaks_pd,
        dotted_lines,
        dotted_lines_1,
        lines_1,
        text_labels,
        text_labels_1,
    )


@app.cell(hide_code=True)
def _(chart_pupu, mo):
    chart_pupu.save('baseline_peak_diffractogram.png')
    download_chart_pupu = mo.download(data="baseline_peak_diffractogram.png", filename="baseline_peak_diffractogram.png", mimetype="image/png", label="Скачать дифрактограмму с общей базовой линией в формате PNG")

    chart_pupu.save('baseline_peak_diffractogram.svg')
    download_chart_pupu_svg = mo.download(data="baseline_peak_diffractogram.svg", filename="baseline_peak_diffractogram.svg", mimetype="image/svg", label="Скачать дифрактограмму с общей базовой линией в формате SVG")

    mo.hstack([download_chart_pupu, download_chart_pupu_svg])
    return download_chart_pupu, download_chart_pupu_svg


@app.cell(hide_code=True)
def _(alt, baseline_gr, df, df_1, df_sel_peaks, graph_height, graph_width):
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
            alt.Chart(df_sel_peaks.to_pandas())
            .mark_rule(strokeWidth=1.5, color='red')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title='')
            )
        )

        # Метки
        text_labels_show = (
            alt.Chart(df_sel_peaks.to_pandas())
            .mark_text(align='left', dx=5, dy=-5, color='black')
            .encode(
                x=alt.X('global_peak_x:Q', title=''),
                y=alt.Y('global_peak_y:Q', title=''),
                text='center_of_mass:N'  # Текст из колонки center_of_mass
            )
        )

    chart_goyfull = (lines_show + baseline_chart + dotted_lines_show + text_labels_show).properties(title=alt.Title(text='Дифрактограмма после обработки со значимыми пиками, их центрами тяжести и ее базовой линией'))
    chart_goyfull
    return (
        baseline_chart,
        chart_goyfull,
        dotted_lines_show,
        lines_show,
        text_labels_show,
    )


@app.cell(hide_code=True)
def _(chart_goyfull, mo):
    chart_goyfull.save('significant_peak_diffractogram.png')
    download_goyfull = mo.download(data="significant_peak_diffractogram.png", filename="significant_peak_diffractogram.png", mimetype="image/png", label="Скачать дифрактограмму со значимыми пиками в формате PNG")

    chart_goyfull.save('significant_peak_diffractogram.svg')
    download_goyfull_svg = mo.download(data="significant_peak_diffractogram.svg", filename="significant_peak_diffractogram.svg", mimetype="image/svg", label="Скачать дифрактограмму со значимыми пиками в формате SVG")

    mo.hstack([download_goyfull, download_goyfull_svg])
    return download_goyfull, download_goyfull_svg


@app.cell(hide_code=True)
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

    char_final = (lines_showff + base_chartiii + dotted_lines_showff + text_labels_showff).properties(title=alt.Title(text='Дифрактограмма после обработки c отдельными базовыми линиями для каждого пика'))
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


@app.cell(hide_code=True)
def _(char_final, mo):
    char_final.save('chart_sepbaseline_peaks.png')
    download_char_final = mo.download(data="chart_sepbaseline_peaks.png", filename="chart_sepbaseline_peaks.png", mimetype="image/png", label="Скачать дифрактограмму с baseline под каждым пиком в формате PNG")

    char_final.save('chart_sepbaseline_peaks.svg')
    download_char_final_svg = mo.download(data="chart_sepbaseline_peaks.svg", filename="chart_sepbaseline_peaks.svg", mimetype="image/png", label="Скачать дифрактограмму с baseline под каждым пиком в формате SVG")

    mo.hstack([download_char_final, download_char_final_svg])
    return download_char_final, download_char_final_svg


@app.cell(hide_code=True)
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
    final_chart = (lines_show_ff + baseline_chart_ff + dotted_lines_show_ff + text_labels_show_ff).properties(title=alt.Title(text='Дифрактограмма после обработки с глобальными пиками, их центрами тяжести и отдельными базовыми линиями на высоте h и 1/2*h'))

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
def _(final_chart, mo):
    final_chart.save('chart_h_sepbaseline_peaks.png')
    download_final_chart = mo.download(data="chart_h_sepbaseline_peaks.png", filename="chart_h_sepbaseline_peaks.png", mimetype="image/png", label="Скачать дифрактограмму с baseline на уровнях h и 1/2*h в формате PNG")

    final_chart.save('chart_h_sepbaseline_peaks.svg')
    download_final_chart_svg = mo.download(data="chart_h_sepbaseline_peaks.svg", filename="chart_h_sepbaseline_peaks.svg", mimetype="image/png", label="Скачать дифрактограмму с baseline на уровнях h и 1/2*h в формате SVG")

    mo.hstack([download_final_chart, download_final_chart_svg])
    return download_final_chart, download_final_chart_svg


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


@app.cell(hide_code=True)
def _(json, mo, pl, table):
    tablet_ret = pl.DataFrame(table.data).to_pandas() # .to_csv(index=False)

    download_tablet_csv = mo.download(
        data=tablet_ret.to_csv(index=False).encode("utf-8"),
        filename="final_table.csv",
        mimetype="text/csv",
        label="Скачать таблицу характеристик в CSV",
    )

    download_tablet_json = mo.download(
        data=json.dumps(tablet_ret.to_json()).encode("utf-8"),
        filename="final_table.json",
        mimetype="application/json",
        label="Скачать таблицу характеристик в JSON",
    )

    mo.hstack([download_tablet_csv, download_tablet_json])
    return download_tablet_csv, download_tablet_json, tablet_ret


@app.cell(hide_code=True)
def _(
    angle_xray,
    global_peaks_window_size,
    graph_height,
    graph_width,
    mo,
    peaks_window_size,
    pl,
    polinom,
    qunatity_peak_kol,
    smooth_polyorder_val,
    smooth_threshold_val,
    smooth_window_size,
    table,
):
    fggg = pl.DataFrame(table.data).to_pandas().to_csv(index=False)

    # Формирование текстового отчета
    report_content = f"""
    Описание переменных:
    - Длина волны: {angle_xray}
    - Размер сглаживающего окна: {smooth_window_size}
    - Уровень значимости: {smooth_threshold_val}
    - Порядок полинома: {smooth_polyorder_val}
    - Количество значений в окрестности основного пика: {global_peaks_window_size}
    - Количество значений в окрестности: {peaks_window_size}
    - Максимальная степень полинома для построения базовой линии: {polinom}
    - Сколько значимых глобальных пиков учитывать: {qunatity_peak_kol}
    - Ширина графика: {graph_width}
    - Высота графика: {graph_height}

    Таблица с характеристиками пиков:
    {fggg}
    """

    # Создание элемента для скачивания отчета
    download_report = mo.download(
        data=report_content.encode("utf-8"),
        filename="report.txt",
        mimetype="text/plain",
        label="Скачать TXT отчет",
    )

    mo.hstack([download_report])
    return download_report, fggg, report_content


if __name__ == "__main__":
    app.run()
