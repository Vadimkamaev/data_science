# разделеение на части по рамеру 'lastactive'
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno
import obrabotka_filtr

# загрузка данных
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    return train, test

# определение среднего размера ошибки
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)

# получение вектора ошибоk
def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * smap

# создание нового таргета
def new_target(raw, param):
    # 'target' ='microbusiness_density' предыдущего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    # 'target' ='microbusiness_density' текущего месяца делить на предыдущего месяца - 1
    raw['target'] = raw['microbusiness_density'] / raw['target'] - 1
    raw.loc[(raw['microbusiness_density'] == 0)|(raw['target'] > 10),'target'] = 0
    raw['target'].fillna(0, inplace=True)

    # raw.loc[(raw['target'] < - 0.0054), 'target'] = - 0.0054
    # raw.loc[(raw['target'] > 0.0054), 'target'] = 0.0054 #
    raw.loc[(raw['target'] > param), 'target'] = param
    raw.loc[(raw['target'] < -param), 'target'] = -param

    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# создание лагов
def build_lag(raw): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 17):
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    #создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
 #   raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1', 'median_hh_inc', 'month']
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки 1.408914 -0.097942
    #maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    maska_train = (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    #maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    maska_val = (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test

# считаем сколько предсказаний лучше
def baz_otchet(raw, mes_1, mes_val):
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[(raw.dcount >= mes_1) & (raw.dcount <= mes_val)].groupby(['cfips', 'dcount'])[['error', 'error_last','lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() >= 0.5
    return len(seria_dt)

# считаем сколько предсказаний лучше после 1-й модели
def otchet(raw, mes_1, mes_val):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    # 'better'=1, если модель лучше чем "=", иначе 'better'=0
    raw['better'] = 0
    raw.loc[raw['error_last'] > raw['error'], 'better'] = 1
    # 'trend_ok'=0 если модель угадала тренд
    raw['trend_ok'] = 0
    maska = (raw['ypred'] >= raw['mbd_lag1']) & (raw['microbusiness_density'] >= raw['mbd_lag1'])
    raw.loc[maska, 'trend_ok'] = 1
    maska = (raw['ypred'] <= raw['mbd_lag1']) & (raw['microbusiness_density'] <= raw['mbd_lag1'])
    raw.loc[maska, 'trend_ok'] = 1
    l = baz_otchet(raw, mes_1, mes_val)
    print('количство cfips предсказанных хуже чем моделью =', l)

# считаем сколько предсказаний лучше после 2-й модели
def model2_otchet(raw, mes_1, mes_val):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l = baz_otchet(raw, mes_1, mes_val)
    print('количство cfips предсказанных хуже чем моделью № 1 =', l)
    raw['error'] = vsmape(raw['microbusiness_density'], raw['y_inegr'])
    l = baz_otchet(raw, mes_1, mes_val)
    print('количство cfips предсказанных хуже чем моделью № 2 =', l)

# вывод на печать информацию по качеству работы модели в каждом штате
def print_state(raw, maska):
    df_state = pd.DataFrame({'state': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for state in raw.state.unique():
        maska2 = (raw['state'] == state) & maska
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (state, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# вывод на печать информацию по качеству работы модели в каждом месяце
def print_month(raw, maska):
    df_state = pd.DataFrame({'month': 0, 'err_mod': 0, 'err_last': 0, 'diff_err': 0, 'lastactive': 0}, index=[0])
    for dcount in raw[maska].dcount.unique():
        maska2 = (raw['dcount'] == dcount) & maska
        target_m = raw.loc[maska2, 'microbusiness_density']
        ypred_m = raw.loc[maska2, 'ypred']
        err_mod1 = smape(target_m, ypred_m)
        mbd_lag11 = raw.loc[maska2, 'mbd_lag1']
        err_last1 = smape(target_m, mbd_lag11)
        st_l_active = raw[maska2]['lastactive'].mean()
        df_state.loc[len(df_state.index)] = (dcount, err_mod1, err_last1, err_last1 - err_mod1, st_l_active)
    #df_state.sort_values(by='diff_err', inplace=True)
    print(df_state.head(53))

# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_test_cfips, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (~raw['cfips'].isin(blac_test_cfips))
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    #print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)
    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    #print('Равенство последнему значению. Ошибка SMAPE:', err_last)

    y_no_blac = raw.loc[maska, 'y_no_blac']
    err_y_no_blac = smape(target, y_no_blac)
    #print('Без блек листа. Ошибка SMAPE:', err_y_no_blac)

    dif_err = err_last - err_mod # положительная - хорошо
    dif_no_blac = err_y_no_blac - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_mod, dif_err, dif_no_blac]
    # print_state(raw, maska)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:', err_mod)
    print('Разность ошибок (чем больше, тем лучше):', dif_err)
    print('Разность с Без блек листа (чем больше, тем лучше):', dif_no_blac)
    return rezult

# модель применяемая в первом процессе оптимизации
def vsia_model(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0):
    # получение трайна и 'y' (игрик) для модели
    df = raw[~raw['cfips'].isin(blac_cfips)]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate=0.0071, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=1,
    )
    # обучение модели
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)
    # прибавляем значимость столбцов новой модели к значениям предыдущих
    znachenie['importance'] = znachenie['importance'] + model.feature_importances_.ravel()
    return y_pred, znachenie

# формирование колонки 'ypred' по результатам 1-го процесса оптимизации
def post_model(raw, y_pred, mes_val, blac_test_cfips):
    # преобразовываем target в 'microbusiness_density'
    y_pred = y_pred + 1
    y_pred = raw.loc[raw.dcount == mes_val, 'mbd_lag1'] * y_pred
    # сохраняем результат обработки одного цикла
    maska =(~raw['cfips'].isin(blac_test_cfips))&(raw.dcount == mes_val)
    raw.loc[maska, 'ypred'] = y_pred
    return raw

# помесячная оптимизация в 1-м процессе
def modeli_po_mesiacam(raw, start_val, stop_model, train_col, blac_cfips, blac_test_cfips):
    start_time = datetime.now()  # время начала работы модели
    znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 2 #
        y_pred, znachenie = vsia_model(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0)
        raw = post_model(raw, y_pred, mes_val, blac_test_cfips)
    #otchet(raw, start_val, mes_val)
    #print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
    # print('Значимость колонок трайна в модели')
    # znachenie.sort_values(by='importance', ascending=False, inplace=True)
    # print(znachenie)
    return raw, blac_cfips

# определяем ошибку модели для сегмента cfips с lastactive в интервале от mini до maxi
# при параметрах min_type=0, min_blac=-1, max_blac=0 определяющих блек лист
def serch_error(raw, rezult, mini, maxi, lastactive, start_val, stop_model, param):
    #metod = 1
    # создаем блек-лист cfips которых не используем в тесте
    maska = (raw['lastactive'] < mini)
    if maxi > 1:
        maska = maska|(raw['lastactive'] >= maxi)
    blac_test_cfips = raw.loc[maska, 'cfips']
    blac_test_cfips = blac_test_cfips.unique()

    # if metod == 0:
    #     #создаем блек-лист cfips который не нужен в трайне
    #     if lastactive >= maxi:
    #         maska = (raw['lastactive'] < mini) | ((raw['lastactive'] > maxi) &
    #                                                  (raw['lastactive'] < lastactive))
    #     elif lastactive <= mini:
    #         maska = (raw['lastactive'] < lastactive)
    #     else:
    #         print('Чо за ерунда?')
    #         maska = (raw['lastactive'] < mini)
    # else:
    maska = (raw['lastactive'] < lastactive)

    blac_cfips = raw.loc[maska, 'cfips']
    blac_cfips = blac_cfips.unique()

    raw = new_target(raw, param)
    # здесь должен начинаться цикл по количеству лагов
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
    # запускаем модель по всем месяцам
    raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_model, train_col, blac_cfips, blac_test_cfips)
    # здесь заканчиваются циклы оптимизации
    return blac_test_cfips, rezult

# распечатываем датафрейм segmenti (УДАЛИТЬ В ОКОНЧАТЕЛЬНОМ ВАРИАНТЕ)
def print_segmenti(segmenti):
    for versia in segmenti['versia'].unique():
        maska = segmenti['versia'] == versia
        df = segmenti[maska]
        print(f"Версия = {versia}. Оптимизированно на {df['optim'].mean()*100}%")
        print(f"'error':{df['error'].mean()}; 'dif_err':{df['dif_err'].mean()}; "
              f"'dif_no_blac':{df['dif_no_blac'].mean()}")
        print(df.head(40))

# для данных с 'lastactive' < minimum делаем предсказание что 'microbusiness_density' не изменилось
def segment_minimum(raw, minimum):
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw.loc[raw['lastactive']<minimum, 'ypred']=raw['mbd_lag1']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# предсказание для отдельного сегмента
def one_fragment(raw, rezult, segmenti, ind, start_val, stop_model, stop_val):
    lastactive = segmenti.loc[ind, 'lastactive']
    mini = segmenti.loc[ind, 'min']
    maxi = segmenti.loc[ind, 'max']
    param = segmenti.loc[ind, 'param']
    if maxi > 1:
        blac_test_cfips, rezult = serch_error(raw, rezult, mini, maxi, lastactive, start_val, stop_model, param)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, lastactive, mini, maxi)
    else:
        blac_test_cfips, rezult = serch_error(raw, rezult, mini, 100000000000, lastactive, start_val, stop_model, param)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, lastactive, mini, maxi)
    return rezult

# предсказываем для всех сегментов в датафрейме segmenti
def vse_fragmenti(raw, rezult, minimum, segmenti):
    #metod = 1 # способ оптимизации
    versia = 1
    start_val = 6  # первый месяц валидации с которого проверяем модель
    stop_model = 39 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 38  # последний месяц валидации до которого проверяем модель
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    df = segmenti[(segmenti['versia'] == versia)]
    for ind, row in df.iterrows(): # цикл по датафрейму сегменты
        # предсказание для отдельного сегмента
        rezult = one_fragment(raw, rezult, segmenti, ind, start_val, stop_model, stop_val)
    raw = segment_minimum(raw, minimum)
    blac_test_cfips = []
    print('Результаты по всей базе')
    rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, 0, 0, 'max')
    otchet(raw, start_val, stop_val)
    # print('Сортировка по dif_no_blac')
    # rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(22))
    return raw

# подсчет кол. элементов в каждом сегменте и запись его в segmenti (УДАЛИТЬ В ОКОНЧАТЕЛЬНОМ ВАРИАНТЕ)
def kol_in_segmenti(raw, segmenti, versia):
    maska = segmenti['versia'] == versia
    df = segmenti[maska]
    for ind, row in df.iterrows():
        maska = raw['lastactive'] >= row['min']
        if row['max'] !=0:
            maska = maska & (raw['lastactive'] < row['max'])
        kol = raw[maska]['cfips'].unique()
        kol = len(kol)
        segmenti.loc[ind,'kol'] = kol
    return segmenti

# создание датафрейма segmenti и сохранение его в файл (УДАЛИТЬ В ОКОНЧАТЕЛЬНОМ ВАРИАНТЕ)
def init_segmentacii(raw):
    granici =[120]
    versia = 1
    d = {'versia':versia,  #возможно несколько версий разбиения в процессе оптимизации
         'min': granici, # 'lastactive' >= 'min'
         'max': granici[1:]+[0], # 'lastactive' < 'max'
         'kol': 0, # количество cfips в интервале
         # параметры оптимизации
         'lastactive': 0, # параметр согласно которому подбираем cfips для трайна
         'param': 0, # граница сглаживания для таргета в трайне
         'optim':0, # 0 - этот интервал от 'min' до 'max' не оптимизирован, 1 - оптимизирован
         'error':0,
         'dif_err':0,
         'dif_no_blac':0}
    segmenti = pd.DataFrame(d)
    segmenti = kol_in_segmenti(raw, segmenti, versia)
    print_segmenti(segmenti)
    segmenti.to_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno3.csv", index=False)

# КОД ПЕРВОГО ПРОЦЕССА ОПТИМИЗАЦИИ ЗАКОНЧЕН. ДАЛЕЕ КОД ПРОЦЕССОВ УВЕЛИЧИВАЮЩИХ ТОЧНОСТЬ ПРЕДСКАЗАНИЯ

# общие функции для 2-х последующих моделей

# создание лагов
def model2_build_lag(raw, param): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 22): # 1.635074     -0.003204          0.014948
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    train_col += ['lastactive', 'state_i']
    return raw, train_col

# модель используемая в методах variant_model и trend_ok_model
def vsia_model2(raw, mes_1, mes_val, train_col, minimum, param=0):
    # получение трайна и 'y' (игрик) для модели 1.637035      0.001173          0.012987
    df = raw[raw.lastactive >= minimum]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate=0.009, #важная штука 1.637233  0.000974  0.012788
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=1,
    )
    # обучение модели
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)
    return y_pred

# Валидация после применения variant_model и trend_ok_model (тестовая, не обязательная функция)
def model2_validacia(raw, start_val, stop_val, rezult, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val)
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_1mod = smape(target, ypred)

    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)

    y_inegr = raw.loc[maska, 'y_inegr']
    err_2mod = smape(target, y_inegr)

    dif_err_1mod = err_1mod - err_2mod # положительная - хорошо
    dif_err_last_mod = err_last - err_2mod  # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_2mod, dif_err_1mod, dif_err_last_mod]

    # print('Предсказано интегрированной моделью. Ошибка SMAPE:', err_2mod)
    # print('Разность ошибок с 1-й моделью (чем больше, тем лучше):', dif_err_1mod)
    # print('Разность ошибок с моделью =:', dif_err_last_mod)
    return rezult

# МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'

# Оптимизация результатов 1-й модели, по результатам модели variant_model (тестовая, не обязательная функция)
# можно удалить в окончательном варианте, т.к. заменяется в main_post_model
def post_variant_model(raw, start_val, stop_model, param):
    # сохраняем результат обработки одного цикла
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    mas0 = (raw['better_pred'] < 0.41)
    maska0 = maska & mas0
    maska1 = maska & (~mas0)
    raw.loc[maska0, 'y_inegr'] = raw.loc[maska0, 'mbd_lag1']*0.4 + raw.loc[maska0, 'ypred']*0.6
    raw.loc[maska1, 'y_inegr'] = raw.loc[maska1, 'ypred']
    return raw

# помесячное предсказание какая из моделей лучше
def variant_model_po_mesiacam(raw, start_val, stop_model, train_col, minimum, param):
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):
        mes_1 = 4 # первый месяц для трайна
        y_pred = vsia_model2(raw, mes_1, mes_val, train_col, minimum, param)
        maska = (raw.dcount == mes_val)
        raw.loc[maska, 'better_pred'] = y_pred
    return raw

# модель предсказывающая какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'
def variant_model(raw, minimum):
    start_val = 10  # первый месяц валидации с которого проверяем модель
    stop_model = 39 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 38  # последний месяц валидации до которого проверяем модель
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    raw['better_pred']=0
    raw['target']=raw['better']
    for param in range(0,1,1):
        raw, train_col = model2_build_lag(raw, param)
        raw = variant_model_po_mesiacam(raw, start_val, stop_model, train_col, minimum, param/10)
        raw = post_variant_model(raw, start_val, stop_model, param)
        rezult = model2_validacia(raw, start_val, stop_val, rezult, param)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        model2_otchet(raw, start_val, stop_val)

# МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ правильно ли предсказан тренд в 1-й модели

# Оптимизация результатов 1-й модели, по результатам модели trend_ok_model (тестовая, не обязательная функция)
# можно удалить в окончательном варианте, т.к. заменяется в main_post_model
def trend_ok_post_model(raw, start_val, stop_model, param):
    # сохраняем результат обработки одного цикла
    maska =(raw.dcount >= start_val)&(raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    mas0 = (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < 910)
    maska0 = maska & mas0
    raw.loc[maska0, 'y_inegr'] = raw.loc[maska0, 'mbd_lag1']
    return raw

# помесячное предсказание тренда
def trend_ok_po_mesiacam(raw, start_val, stop_model, train_col, minimum, param):
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 4 # первый месяц для трайна
        y_pred = vsia_model2(raw, mes_1, mes_val, train_col, minimum, param)
        maska = (raw.dcount == mes_val)
        raw.loc[maska, 'trend_ok_pred'] = y_pred
    return raw

# модель предсказывающая правильно ли предсказан тренд в 1-й модели
def trend_ok_model(raw, minimum):
    start_val = 10  # первый месяц валидации с которого проверяем модель
    stop_model = 39 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 38  # последний месяц валидации до которого проверяем модель
    rezult = pd.DataFrame(columns=['lastactive', 'param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    validacia(raw, start_val, stop_val, rezult, [], 0, 0, 0)
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    raw['trend_ok_pred']=0
    raw['target']=raw['trend_ok']
    for param in range(1,1,1):
        raw, train_col = model2_build_lag(raw, param)
        raw = trend_ok_po_mesiacam(raw, start_val, stop_model, train_col, minimum, param)
        raw = trend_ok_post_model(raw, start_val, stop_model, param)
        rezult = model2_validacia(raw, start_val, stop_val, rezult, param)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        model2_otchet(raw, start_val, stop_val)
    return raw

# КОД МОДЕЛЕЙ ЗАКОНЧЕН. ДАЛЕЕ ОПРЕДЕЛЯЕМ В КАКОМ СЛУЧАЕ КАКУЮ МОДЕЛЬ ПРИМЕНЯТЬ

def state_post_model05(raw, state, start_val, stop_model, param):
    mean_porog = 910
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    m_s = raw['state'] == state
    maska_state = maska & m_s
    maska_over = maska & (~m_s)
    maska_over = maska_over & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < 910)
    raw.loc[maska_over, 'y_inegr'] = raw.loc[maska_over, 'mbd_lag1']
    maska_state = maska_state & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < param)
    raw.loc[maska_state, 'y_inegr'] = raw.loc[maska_state, 'mbd_lag1']
    return raw

def trend05_post_model(raw, df_state, minimum, start_val, stop_model, stop_val):
    for state in raw['state'].unique():
        rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
        param = minimum
        while param < 50000:
            raw = state_post_model05(raw, state, start_val, stop_model, param)  # 1.633834  0.000187 0.016188
            rezult = model2_validacia(raw, start_val, stop_val, rezult, state, param)
            param = round(param * 1.02)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        param2 = rezult.iloc[0]['param2']
        err_2mod = rezult.iloc[0]['err_2mod']
        df_state.loc[len(df_state.index)] = [state, 0, param2, 0, 0, err_2mod]
        print(df_state.head(60))

def state_post_model_granica(raw, df_state, state, start_val, stop_model, param):
    trend05 = df_state.loc[df_state['state']==state, 'trend05']
    trend05 = trend05.mean()
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    m_s = raw['state'] == state
    maska_state = maska & m_s
    maska_over = maska & (~m_s)
    maska_over = maska_over & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < 910)
    raw.loc[maska_over, 'y_inegr'] = raw.loc[maska_over, 'mbd_lag1']

    maska05 = maska_state & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < trend05)
    raw.loc[maska05, 'y_inegr'] = raw.loc[maska05, 'mbd_lag1']

    maska_gran = maska_state & (raw['lastactive'] < param)
    raw.loc[maska_gran, 'y_inegr'] = raw.loc[maska_gran, 'mbd_lag1']
    # maska03 = maska_state & (raw['trend_ok_pred'] < 0.3)&(raw['lastactive'] < param)
    # raw.loc[maska03, 'y_inegr'] = raw.loc[maska03, 'mbd_lag1']
    return raw

def granica_post_model(raw, df_state, minimum, start_val, stop_model, stop_val):
    for state in raw['state'].unique():
        #trend05 = df_state.loc[df_state['state']==state, 'trend05']
        rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
        param = minimum
        while param <= 50000:#trend05[0]:
            raw = state_post_model_granica(raw, df_state, state, start_val, stop_model, param)
            rezult = model2_validacia(raw, start_val, stop_val, rezult, state, param)
            param = round(param * 1.02)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        param2 = rezult.iloc[0]['param2']
        err_2mod = rezult.iloc[0]['err_2mod']
        df_state.loc[df_state['state']==state, 'granica'] = param2
        print(df_state.head(60))

def state_post_model03(raw, df_state, state, start_val, stop_model, param):
    trend05 = df_state.loc[df_state['state']==state, 'trend05']
    trend05 = trend05.mean()

    granica = df_state.loc[df_state['state'] == state, 'granica']
    granica = granica.mean()

    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    m_s = raw['state'] == state
    maska_state = maska & m_s
    maska_over = maska & (~m_s)
    maska_over = maska_over & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < 910)
    raw.loc[maska_over, 'y_inegr'] = raw.loc[maska_over, 'mbd_lag1']

    maska05 = maska_state & (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < trend05)
    raw.loc[maska05, 'y_inegr'] = raw.loc[maska05, 'mbd_lag1']

    maska_gran = maska_state & (raw['lastactive'] < granica)
    raw.loc[maska_gran, 'y_inegr'] = raw.loc[maska_gran, 'mbd_lag1']

    maska03 = maska_state & (raw['trend_ok_pred'] < 0.3)&(raw['lastactive'] < param)
    raw.loc[maska03, 'y_inegr'] = raw.loc[maska03, 'mbd_lag1']
    return raw

def trend03_post_model(raw, df_state, minimum, start_val, stop_model, stop_val):
    for state in raw['state'].unique():
        rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
        param = minimum
        while param < 50000:
            state_post_model03(raw, df_state, state, start_val, stop_model, param)
            rezult = model2_validacia(raw, start_val, stop_val, rezult, state, param)
            param = round(param * 1.02)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        param2 = rezult.iloc[0]['param2']
        err_2mod = rezult.iloc[0]['err_2mod']
        df_state.loc[df_state['state']==state, 'trend03'] = param2
        print(df_state.head(60))

def work_state(raw, df_state, minimum, start_val, stop_model, stop_val):
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    for i, stat in df_state.iterrows():
        trend03 = stat['trend03']
        trend05 = stat['trend05']
        granica = stat['granica']
        state = stat['state']
        maska_state = maska & (raw['state'] == state)
        maska05 = maska_state & (raw['trend_ok_pred'] < 0.5) & (raw['lastactive'] < trend05)
        raw.loc[maska05, 'y_inegr'] = raw.loc[maska05, 'mbd_lag1']

        maska_gran = maska_state & (raw['lastactive'] < granica)
        raw.loc[maska_gran, 'y_inegr'] = raw.loc[maska_gran, 'mbd_lag1']

        maska03 = maska_state & (raw['trend_ok_pred'] < 0.3) & (raw['lastactive'] < trend03)
        raw.loc[maska03, 'y_inegr'] = raw.loc[maska03, 'mbd_lag1']
    rezult = model2_validacia(raw, start_val, stop_val, rezult)
    print(rezult.head(22))

# Выбор какую из моделей в каком случае применять на основе
def main_post_model(raw, minimum):
    start_val = 10  # первый месяц валидации с которого проверяем модель
    stop_model = 39 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 38  # последний месяц валидации до которого проверяем модель

    # БЛОК ОПТИМИЗАЦИИ ПОСТ-МОДЕЛИ С ФОМИРОВАНИЕМ df_state

    # df_state = pd.DataFrame(columns=['state', 'granica', 'trend05', 'trend03',  'gran_miks', 'error'])
    # trend05_post_model(raw, df_state, minimum, start_val, stop_model, stop_val)
    # df_state.to_csv("C:\\kaggle\\МикроБизнес\\state.csv", index=False)

    # df_state = pd.read_csv("C:\\kaggle\\МикроБизнес\\state.csv")
    # granica_post_model(raw, df_state, minimum, start_val, stop_model, stop_val)
    # df_state.to_csv("C:\\kaggle\\МикроБизнес\\state.csv", index=False)

    # df_state = pd.read_csv("C:\\kaggle\\МикроБизнес\\state.csv")
    # trend03_post_model(raw, df_state, minimum, start_val, stop_model, stop_val)
    # df_state.to_csv("C:\\kaggle\\МикроБизнес\\state.csv", index=False)

    df_state = pd.read_csv("C:\\kaggle\\МикроБизнес\\state.csv")
    work_state(raw, df_state, minimum, start_val, stop_model, stop_val)

    model2_otchet(raw, start_val, stop_val)

def otvet(raw):
    test = raw[raw.istest == 1].copy()
    test['microbusiness_density'] = test['ypred']
    test = test[['row_id', 'cfips', 'microbusiness_density']]
    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\main0.csv', index=False)

def model2_otvet(raw):
    test = raw[raw.istest == 1].copy()
    test['microbusiness_density'] = test['y_inegr']
    test = test[['row_id', 'cfips', 'microbusiness_density']]
    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\model2.csv', index=False)

if __name__ == "__main__":
    minimum = 60 # ниже этого значения 'lastactive' считаем что 'microbusiness_density' не меняется
    pd.options.display.width = 0 # для печати
    # train, test = start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)

    # датафрейм для хранения результатов вариантов оптимизации
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])

    # ЗАГРУЗКА ДАННЫХ
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_no_blac.csv")
    # raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    # raw['ypred'] = 0

    # ПЕРВЫЙ ПРОЦЕСС ОПТИМИЗАЦИИ
    # segmenti = pd.read_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno2.csv")
    # raw = vse_fragmenti(raw, rezult, minimum, segmenti) # первая модель
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_posle1modeli2.csv", index=False)
    # otvet(raw)

    # МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_posle1modeli2.csv")
    raw['y_inegr'] = 0
    variant_model(raw, minimum) #модель предсказывающая ошибку определения тренда первой моделью
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_posle2modeli.csv", index=False)

    # МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ правильно ли предсказан тренд в 1-й модели
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_posle1modeli2.csv")
    # raw['y_inegr'] = 0
    # trend_ok_model(raw, minimum) #модель предсказывающая ошибку определения тренда первой моделью
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_posle2modeli.csv", index=False)

    # ОПРЕДЕЛЯЕМ В КАКОМ СЛУЧАЕ КАКУЮ МОДЕЛЬ ПРИМЕНЯТЬ

    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_posle2modeli.csv")
    # main_post_model(raw, minimum) # выбираем первую модель или модель не изменения 'microbusiness_density'

    # СОЗДАЕМ ФАЙЛ С РЕЗУЛЬТАТОМ
    # model2_otvet(raw)
