import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

def nevedomo(train):
    # SET VALIDATE = TRUE ДЛЯ ВЫЧИСЛЕНИЯ ПРОВЕРКИ.
    # И УСТАНОВИТЕ VALIDATE = FALSE ДЛЯ ПОДАЧИ В LB
    VALIDATE = True
    # ИСПОЛЬЗУЙТЕ 1 ДЛЯ ПРОВЕРКИ ПОСЛЕДНЕГО МЕСЯЦА ПОЕЗДА, 2 ДЛЯ ПОСЛЕДНЕГО И Т.Д. И Т.Д.
    # НОВЫЕ ДАННЫЕ О ПОЕЗДЕ ДО ЭТОГО
    VAL_MONTH = 1
    # DEFINE VALIDATION AND TRAIN MONTHS
    trn_months = train.first_day_of_month.values[-39:-1 * VAL_MONTH] # 37 первых дней месяцев в массиве
    val_months = train.first_day_of_month.values[-1 * VAL_MONTH:] # 1-й день последнего месяца
    if VAL_MONTH != 1:
        val_months = train.first_day_of_month.values[-1 * VAL_MONTH:-1 * VAL_MONTH + 1]

    # CREATE NEW TRAIN AND NEW TEST DATA IF WE ARE VALIDATING
    if VALIDATE:
        test = train.loc[train.first_day_of_month.isin(val_months)] # фильтр по дате val_months (посл. дата)
        train = train.loc[train.first_day_of_month.isin(trn_months)] # фильтр по остальным датам
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

# Обучите и выведите сезонную модель
# Для каждого из 3135 временных рядов мы смотрим на последние 5 microbusiness_density.
# Мы также смотрим на эти 5 месяцев 1 года назад. А мы смотрим на 2 года назад.
# Если мы увидим восходящий тренд для всех трех временных рядов длиной 5, то мы прогнозируем последнее значение,
# умноженное на 1,003. Если мы видим нисходящий тренд, мы прогнозируем последнее значение, умноженное на 0,997.
# Мы корректируем последнее значение только для округов с населением более 35 000 человек
# (или любое другое значение, которое мы объявляем в ACTIVE_THRESHOLD).
def sezon_bred(train, test):
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    VALIDATE = False
    TRAIN_SZ = len(train) // 3135
    TEST_SZ = len(test) // 3135
    DISPLAY = 8
    # ПОРОГ ОШИБКИ В ПРОЦЕНТАХ = THRESHOLD / 78
    THRESHOLD = 8  # для линейной регрессии
    ACTIVE_THRESHOLD = 9_000

    IDS = train.cfips.unique() # массив уникальных cfips
    x_train = np.arange(TRAIN_SZ).reshape((-1, 1))
    x_test = np.arange(TRAIN_SZ - 1, TRAIN_SZ + TEST_SZ).reshape((-1, 1))

    linear_preds = np.zeros((len(IDS), TEST_SZ))
    last_preds = np.zeros((len(IDS), TEST_SZ))
    seasonal_preds = np.zeros((len(IDS), TEST_SZ))
    sn_trend = 0
    lin_trend = 0

    ct = 0
    for i, c in enumerate(IDS): #цикл по уникальным cfips
        df = train.loc[train.cfips == c] # датафрейм отфильтрованный по cfips
        last = df.microbusiness_density.values[-1] # последнее значение
        active = df.active.values[-1] # последнее значение
        last_preds[i,] = [last] * TEST_SZ

        # ВЫЧИСЛЕНИЕ СЕЗОННЫХ ТЕНДЕНЦИЙ
        WIDTH1 = 5
        WIDTH2 = 7
        WIDTH3 = 7
        # НЕДАВНИЙ
        x0a = df.microbusiness_density.values[-1 - WIDTH1:-1]
        x0 = np.median(x0a)
        # 1 год назад
        x1 = df.microbusiness_density.values[-12 - 1 + 1]
        x2a = df.microbusiness_density.values[-12 - 1 - WIDTH2 + 1:-12 - 1 + 1]
        x2 = np.median(x2a)
        # 2 года назад
        x3 = df.microbusiness_density.values[-24 - 1 + 1]
        x4a = df.microbusiness_density.values[-24 - 1 - WIDTH3 + 1:-24 - 1 + 1]
        x4 = np.median(x4a)

        # FIT TRANSFORM SEASONAL MODEL
        p = last # последнее значение microbusiness_density
        if active >= ACTIVE_THRESHOLD:
            if (x1 > x2) & (x3 > x4) & (last > x0):
                p *= 1.003
            elif (x1 < x2) & (x3 < x4) & (last < x0):
                p *= 0.997
        seasonal_preds[i,] = [p] * TEST_SZ

        # FIT TRANSFORM LINEAR REGRESSION MODEL
        model = LinearRegression()
        model.fit(x_train, df.microbusiness_density)
        p = model.predict(x_train)
        err = p - df.microbusiness_density.values
        rng = df.microbusiness_density.max() - df.microbusiness_density.min()

        # ОПРЕДЕЛИТЕ, ЯВЛЯЕТСЯ ЛИ ВРЕМЕННОЙ РЯД ЛИНЕЙНЫМ ИЛИ НЕТ
        s = 0
        for k in range(TRAIN_SZ):
            e = np.abs(err[k])
            r = e / (rng / 2)
            s += r
        if (s > THRESHOLD) | (active < ACTIVE_THRESHOLD):
            linear_preds[i,] = [last] * TEST_SZ
        else:
            p2 = model.predict(x_test)
            shift = last - p2[0]
            linear_preds[i,] = p2[1:] + shift
            lin_trend += 1

        # СЧИТАТЬ МАТЕРИАЛЫ
        if seasonal_preds[i,][0] == last: continue
        ct += 1
        sn_trend += 1
        if ct >= DISPLAY + 1: continue

        mid = 'downward '
        if seasonal_preds[i,][0] > last: mid = 'upward '
    # НАПЕЧАТАЙТЕ, СКОЛЬКО ЛИНЕЙНЫХ ВРЕМЕННЫХ РЯДОВ МЫ НАШЛИ
    print(f'There are {lin_trend} counties with both a linear trend and large population.')
    print(f'There are {sn_trend} counties with both a seasonal trend and large population.')

    #def prodolshenie_breda():
    # ПРИМЕЧАНИЕ ТЕСТ СОРТИРОВАН ПО CFIPS И ДАННЫМ. ДОБАВИТЬ PREDS В ДАННЫЕ
    if VALIDATE: test['true'] = test.microbusiness_density.copy()
    test['last'] = last_preds.reshape((-1))
    test['linear'] = linear_preds.reshape((-1))
    test['seasonal'] = seasonal_preds.reshape((-1))

    # ENSEMBLE SEASONAL AND LINEAR
    d1 = test.seasonal != test['last']
    d2 = test.linear != test['last']
    print((d1 | d2).sum())
    test['microbusiness_density'] = (test.seasonal + test.linear) / 2.
    test.loc[d1 & (~d2), 'microbusiness_density'] = test.loc[d1 & (~d2), 'seasonal']
    test.loc[(~d1) & d2, 'microbusiness_density'] = test.loc[(~d1) & d2, 'linear']

    # CREATE SUBMISSION CSV
    sub = test[['row_id', 'microbusiness_density']]
    if not VALIDATE:
        sub.to_csv('C:\\kaggle\\МикроБизнес\\submiss123.csv', index=False)
    else:
        sub.to_csv('C:\\kaggle\\МикроБизнес\\validation.csv', index=False)




train, test0 = start()
train, test = nevedomo(train)
sezon_bred(train, test0)