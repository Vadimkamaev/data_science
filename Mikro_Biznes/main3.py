import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno
# import obrabotka_filtr
# from scipy.optimize import Bounds, minimize

# загрузка данных
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    census = pd.read_csv("C:\\kaggle\\МикроБизнес\\census_starter.csv")
    new_data = pd.read_csv("C:\\kaggle\\МикроБизнес\\revealed_test.csv")
    return train, test, census, new_data

# загрузка данных переписи
def load_perepis():
    # загузка из ноутбука
    # df2020 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2020.S0101-Data.csv', usecols=COLS)
    # df2021 = pd.read_csv('/kaggle/input/census-data-for-godaddy/ACSST5Y2021.S0101-Data.csv',usecols=COLS)
    df2020 = pd.read_csv("C:\\kaggle\\МикроБизнес\\ACSST5Y2020.S0101-Data.csv", dtype = 'object')
    df2021 = pd.read_csv("C:\\kaggle\\МикроБизнес\\ACSST5Y2021.S0101-Data.csv", dtype = 'object')

    df2020 = df2020[['GEO_ID', 'S0101_C01_026E']]
    df2020 = df2020.iloc[1:]
    # df2020['S0101_C01_026E'] = df2020['S0101_C01_026E'].astype('int')
    df2020 = df2020.astype({'S0101_C01_026E':'int'})

    df2021 = df2021[['GEO_ID', 'S0101_C01_026E']]
    df2021 = df2021.iloc[1:]
    #df2021['S0101_C01_026E'] = df2021['S0101_C01_026E'].astype('int')
    df2021 = df2021.astype({'S0101_C01_026E':'int'})


    df2020['cfips'] = df2020.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
    adult2020 = df2020.set_index('cfips').S0101_C01_026E.to_dict()


    df2021['cfips'] = df2021.GEO_ID.apply(lambda x: int(x.split('US')[-1]))
    adult2021 = df2021.set_index('cfips').S0101_C01_026E.to_dict()

    # df2020 = df2020[['cfips','S0101_C01_026E']]
    # df2021 = df2021[['cfips','S0101_C01_026E']]

    # df2020['adult2021'] = df2021['S0101_C01_026E']
    # df2020['mnoshitel'] = df2020['S0101_C01_026E']/df2020['adult2021']

    df2020['mnoshitel'] = df2020['S0101_C01_026E'] / df2021['S0101_C01_026E']

    df2020 = df2020[['cfips','mnoshitel']]
    df2020.set_index('cfips', inplace=True)

    # ser2020 = sub.cfips.map(adult2020)
    # ser2021 = sub.cfips.map(adult2021)
    # mnoshitel = ser2020 / ser2021
    return df2020


# объединенный массив трейна и теста, создание объединенного raw
def maceraw(train, test):
    train['istest'] = 0
    test['istest'] = 1
    # объединенный массив трейна и теста
    raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year
    raw["month"] = raw["first_day_of_month"].dt.month
    # перенумерация внутри группы - номер месяца
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
    # кодирование целыми числами
    #raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    return raw

# добавление в raw данных за ноябрь и декабрь
def new_data_in_raw(raw, new_data):
    for index, row_new in new_data.iterrows():
        maska = row_new['row_id'] == raw['row_id']
        raw.loc[maska, 'microbusiness_density'] = row_new['microbusiness_density']
        raw.loc[maska, 'active'] = row_new['active']
    return raw

# обработка файла census
def censusdef(census, raw):
    columns = ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']
    raw[columns] = 0
    for index, row_census in census.iterrows():
        for year in range(2017, 2022):
            for col in columns:
                ncol = col+'_'+str(year)
                maska = (raw['year']==year+2)&(raw['cfips']==row_census['cfips'])
                raw.loc[maska, col] = row_census[ncol]
    return raw

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

    # raw.loc[(raw['target'] < - 0.0074), 'target'] = - 0.0074
    # raw.loc[(raw['target'] > 0.0074), 'target'] = 0.0074  #
    raw.loc[(raw['target'] > param), 'target'] = param
    raw.loc[(raw['target'] < -param), 'target'] = -param

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
def baz_otchet(raw):
    global start_val, stop_val, minimum
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw.lastactive > minimum)
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[maska].groupby(['cfips', 'dcount'])[['error', 'error_last','lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() >= 0.5

    hor = (raw.loc[maska, 'error_last'] > raw.loc[maska, 'error']).sum()
    ploh = (raw.loc[maska, 'error'] > raw.loc[maska, 'error_last']).sum()
    notdif = len(raw[maska]) - hor - ploh

    return len(seria_dt), ploh, hor, notdif

# считаем сколько предсказаний лучше после 1-й модели
def otchet(raw):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l, ploh, hor, notdif = baz_otchet(raw)
    raw['better'] = 0 # 'better'=1, если модель лучше чем "=", иначе 'better'=0
    raw.loc[raw['error_last'] > raw['error'], 'better'] = 1
    # 'trend_ok'=0 если модель угадала тренд
    raw['trend_ok'] = 0
    maska = (raw['ypred'] >= raw['mbd_lag1']) & (raw['microbusiness_density'] >= raw['mbd_lag1'])
    raw.loc[maska, 'trend_ok'] = 1

    maska = (raw['ypred'] <= raw['mbd_lag1']) & (raw['microbusiness_density'] <= raw['mbd_lag1'])
    raw.loc[maska, 'trend_ok'] = 1
    print('количство cfips предсказанных хуже чем моделью =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)

# считаем сколько предсказаний лучше после 2-й модели
def model2_otchet(raw):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    l, ploh, hor, notdif = baz_otchet(raw)
    print('')
    print('------ Отчет после выполнения оптимизации 2-й модели ------')
    print('количство cfips предсказанных хуже чем моделью № 1 =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)
    raw['error'] = vsmape(raw['microbusiness_density'], raw['y_inegr'])
    l, ploh, hor, notdif = baz_otchet(raw)
    print('количство cfips предсказанных хуже чем моделью № 2 =', l,
          'Отношение плохих предсказаний к хорошим', ploh/hor)
    print('Кол. хороших', hor, 'Плохих', ploh, 'Равных', notdif)

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

# тагрет для проверки модели. Убираем из проверки выбросы более чем granica
def target_for_error(raw, maska):
    # granica = 0.2
    target = raw.loc[maska, 'microbusiness_density']
    # m = raw['microbusiness_density'] > (1 + granica) * raw['mbd_lag1']
    # target.loc[m] = raw.loc[maska & m, 'mbd_lag1'] * (1 + granica)
    # m = raw['microbusiness_density'] < (1 - granica) * raw['mbd_lag1']
    # target.loc[m] = raw.loc[maska & m, 'mbd_lag1'] * (1 - granica)
    return target

# Валидация
def validacia(raw, rezult, blac_test_cfips, param1=1, param2=1, param3=1):
    global  start_val, stop_val
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (~raw['cfips'].isin(blac_test_cfips))
    target = target_for_error(raw, maska)
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)

    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)

    # y_no_blac = raw.loc[maska, 'y_no_blac']
    # err_y_no_blac = smape(target, y_no_blac)

    dif_err = err_last - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_mod, dif_err, 0]

    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:', err_mod)
    print('Разность ошибок (чем больше, тем лучше):', dif_err)
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
        n_jobs=3,
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
    mask = raw.dcount == mes_val
    # преобразовываем target в 'microbusiness_density'
    y_pred = y_pred + 1
    y_pred = raw.loc[mask, 'mbd_lag1'] * y_pred
    # сохраняем результат обработки одного цикла
    maska =(~raw['cfips'].isin(blac_test_cfips))&(raw.dcount == mes_val)
    raw.loc[maska, 'ypred'] = y_pred
    return raw

# помесячная оптимизация в 1-м процессе
def modeli_po_mesiacam(raw, train_col, blac_cfips, blac_test_cfips):
    global start_val, stop_model
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
def serch_error(raw, rezult, mini, maxi, lastactive, param):
    # создаем блек-лист cfips которых не используем в тесте
    maska = (raw['lastactive'] < mini)
    if maxi > 1:
        maska = maska|(raw['lastactive'] >= maxi)
    blac_test_cfips = raw.loc[maska, 'cfips']
    blac_test_cfips = blac_test_cfips.unique()

    maska = (raw['lastactive'] < lastactive)

    blac_cfips = raw.loc[maska, 'cfips']
    blac_cfips = blac_cfips.unique()

    raw = new_target(raw, param)
    # здесь должен начинаться цикл по количеству лагов
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
    # запускаем модель по всем месяцам
    raw, blac_cfips = modeli_po_mesiacam(raw, train_col, blac_cfips, blac_test_cfips)
    # здесь заканчиваются циклы оптимизации
    return blac_test_cfips, rezult

# распечатываем датафрейм segmenti (УДАЛИТЬ В ОКОНЧАТЕЛЬНОМ ВАРИАНТЕ)
def print_segmenti(segmenti):
    for versia in segmenti['versia'].unique():
        maska = segmenti['versia'] == versia
        df = segmenti[maska]
        print(f"Версия = {versia}. Оптимизированно на {df['optim'].mean()*100}%")
        print(f"'error':{df['error'].mean()}; 'dif_err':{df['dif_err'].mean()}")
        print(df.head(40))

# для данных с 'lastactive' < minimum делаем предсказание что 'microbusiness_density' не изменилось
def segment_minimum(raw):
    global minimum
    # raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw.loc[raw['lastactive']<minimum, 'ypred']=raw['mbd_lag1']
    raw['ypred'].fillna(0, inplace=True)
    return raw

# предсказание для отдельного сегмента
def one_fragment(raw, rezult, segmenti, ind, stop_val):
    lastactive = segmenti.loc[ind, 'lastactive']
    mini = segmenti.loc[ind, 'min']
    maxi = segmenti.loc[ind, 'max']
    param = segmenti.loc[ind, 'param']
    print ('Обрабатываем фрагмент от', mini, 'до', maxi)
    if maxi > 1:
        blac_test_cfips, rezult = serch_error(raw, rezult, mini, maxi, lastactive, param)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, rezult, blac_test_cfips, lastactive, mini, maxi)
    else:
        blac_test_cfips, rezult = serch_error(raw, rezult, mini, 100000000000, lastactive, param)
        # валидация по результату обработки всех месяцев валидации в цикле
        rezult = validacia(raw, rezult, blac_test_cfips, lastactive, mini, maxi)
    return rezult

# предсказываем для всех сегментов в датафрейме segmenti
def model1_vse_fragmenti(raw, rezult, segmenti):
    #metod = 1 # способ оптимизации
    versia = 1
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    df = segmenti[(segmenti['versia'] == versia)]
    for ind, row in df.iterrows(): # цикл по датафрейму сегменты
        # предсказание для отдельного сегмента
        rezult = one_fragment(raw, rezult, segmenti, ind, stop_val)
    raw = segment_minimum(raw)
    blac_test_cfips = []
    print('')
    print('------ Отчет после выполнения оптимизации 1-й модели ------')
    print('Результаты по всей базе')
    rezult = validacia(raw, rezult, blac_test_cfips, 0, 0, 'max')
    otchet(raw)
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
def vsia_model2(raw, mes_1, mes_val, train_col, param=0):
    global minimum
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
        n_jobs=3,
    )
    # обучение модели
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)
    return y_pred

# Валидация после применения variant_model и trend_ok_model (тестовая, не обязательная функция)
def model2_validacia(raw, rezult, param1=1, param2=1, param3=1):
    global start_val, stop_val
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val)
    target = target_for_error(raw, maska)
    ypred = raw.loc[maska, 'ypred']
    err_1mod = smape(target, ypred)

    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)

    y_inegr = raw.loc[maska, 'y_inegr']
    err_2mod = smape(target, y_inegr)

    dif_err_1mod = err_1mod - err_2mod # положительная - хорошо
    dif_err_last_mod = err_last - err_2mod  # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_2mod, dif_err_1mod, dif_err_last_mod]

    print('Предсказано интегрированной моделью. Ошибка SMAPE:', err_2mod)
    print('Разность ошибок с 1-й моделью (чем больше, тем лучше):', dif_err_1mod)
    print('Разность ошибок с моделью =:', dif_err_last_mod)
    return rezult

# МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'

# Оптимизация результатов 1-й модели, по результатам модели variant_model (тестовая, не обязательная функция)
# можно удалить в окончательном варианте, т.к. заменяется в main_post_model
def post_variant_model(raw, param):
    global start_val, stop_model
    # сохраняем результат обработки одного цикла
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw['y_inegr']=raw['ypred']
    l=500
    m=-0.6
    k = raw.loc[maska, 'lastactive']/l + m
    fig = (raw.loc[maska, 'better_pred']+k).clip(0,1)
    maska = maska & (raw.ypred != raw.mbd_lag1)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'mbd_lag1'] * (1 - fig) + raw.loc[maska, 'ypred'] * fig
    return raw

# помесячное предсказание какая из моделей лучше
def variant_model_po_mesiacam(raw, train_col, param):
    global start_val, stop_model
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):
        mes_1 = 4 # первый месяц для трайна
        y_pred = vsia_model2(raw, mes_1, mes_val, train_col, param)
        maska = (raw.dcount == mes_val)
        raw.loc[maska, 'better_pred'] = y_pred
    return raw

# модель предсказывающая какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'
def variant_model(raw):
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    raw['better_pred']=0 # 0 (лучше модель =) до 1 (лучше 1-я модель)
    raw['target']=raw['better']
    for param in range(0,1,1):
        raw, train_col = model2_build_lag(raw, param)
        raw = variant_model_po_mesiacam(raw, train_col, param/10)
        raw = post_variant_model(raw, param)
        print('')
        print('------ variant_model ---------')
        rezult = model2_validacia(raw, rezult, param)
        print('Сортировка по err_2mod')
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        # model2_otchet(raw)

# МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ правильно ли предсказан тренд в 1-й модели

# Оптимизация результатов 1-й модели, по результатам модели trend_ok_model (тестовая, не обязательная функция)
# можно удалить в окончательном варианте, т.к. заменяется в main_post_model
def trend_ok_post_model(raw, param1, param2):
    global start_val, stop_model
    # сохраняем результат обработки одного цикла
    maska =(raw.dcount >= start_val)&(raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    # mas0 = (raw['trend_ok_pred'] < 0.5)&(raw['lastactive'] < 910)
    mas0 = (raw['trend_ok_pred']*param1 + raw['lastactive']) < param2
    maska0 = maska & mas0
    raw.loc[maska0, 'y_inegr'] = raw.loc[maska0, 'mbd_lag1']
    return raw

# помесячное предсказание тренда
def trend_ok_po_mesiacam(raw, train_col, param):
    global start_val, stop_model
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_model + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 4 # первый месяц для трайна
        y_pred = vsia_model2(raw, mes_1, mes_val, train_col, param)
        maska = (raw.dcount == mes_val)
        raw.loc[maska, 'trend_ok_pred'] = y_pred
    return raw

# модель предсказывающая правильно ли предсказан тренд в 1-й модели
def trend_ok_model(raw):
    rezult = pd.DataFrame(columns=['lastactive', 'param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    # validacia(raw, start_val, stop_val, rezult, [], 0, 0, 0)
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    raw['trend_ok_pred']=0
    raw['target']=raw['trend_ok']
    for param in range(1,2,1):
        raw, train_col = model2_build_lag(raw, param)
        raw = trend_ok_po_mesiacam(raw, train_col, param)
        # error = optim([240, 400], raw)
        # print('error optim', error)
        # raw = trend_ok_post_model(raw, param)
        print('')
        print('------ trend_ok_model ---------')
        rezult = model2_validacia(raw, rezult, param)
        # print('Сортировка по err_2mod')
        # rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        # print(rezult.head(22))
        # model2_otchet(raw)
    return raw

# КОД МОДЕЛЕЙ ЗАКОНЧЕН. ДАЛЕЕ ОПРЕДЕЛЯЕМ В КАКОМ СЛУЧАЕ КАКУЮ МОДЕЛЬ ПРИМЕНЯТЬ

# для каждого штата в отдельности улучшаем предсказание на основе variant_model
def state_post_var_model(raw, maska_state, param1, param2):
    k = raw.loc[maska_state, 'lastactive'] / param1 + param2
    fig = (raw.loc[maska_state, 'better_pred'] + k).clip(0.5, 1)
    raw.loc[maska_state, 'y_inegr'] = raw.loc[maska_state, 'mbd_lag1'] * (1 - fig) + \
                                      raw.loc[maska_state, 'ypred'] * fig
    return raw

# для каждого штата в отдельности улучшаем предсказание на основе variant_model
def state_post_variant_model(raw, state, param1, param2):
    global start_val, stop_model, minimum
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model) & (raw.lastactive > minimum)
    maska_state = maska & (raw['state'] == state)
    raw = state_post_var_model(raw, maska_state, param1, param2)
    return raw

# ошибка 2-й модели - variant_model и
def model2_error(raw, state):
    global start_val, stop_val, minimum
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw['state'] == state) & \
            (raw.lastactive > minimum)
    target = target_for_error(raw, maska)
    y_inegr = raw.loc[maska, 'y_inegr']
    error = smape(target, y_inegr)
    return error

# функция, которая оптимизируется. Используется в variant_model и trend_ok
def optim_variant(param1, param2, raw, state):
    state_post_variant_model(raw, state, param1, param2)
    r = model2_error(raw, state)
    raw['y_inegr'] = raw['y_prom']
    return r

#Поиск широкого минимума в прямоугольнике. Используется в variant_model и trend_ok
def min_in_kvadrat(x1, y1, shag1, shag2, delit_x, delit_y, raw, state, optim):
    global start_val, stop_model, stop_val
    maska =(raw.dcount >= start_val)&(raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']
    min_err = 1000  #model2_error(raw, state)
    matrica = np.zeros((delit_x, delit_y))
    for i in tqdm(range(delit_x)):
        for j in range(delit_y):
            param1 = x1 + i * shag1
            param2 = y1 + j * shag2
            error = optim(param1, param2, raw, state)
            matrica[i,j] = error
    for i in range(1, delit_x-1):
        for j in range(1, delit_y-1):
            error = matrica[i,j]*4
            error += (matrica[i-1,j]+matrica[i+1,j]+matrica[i,j-1]+matrica[i,j+1])*2
            error += matrica[i-1, j-1] + matrica[i+1, j+1 ] + matrica[i+1, j-1] + matrica[i-1, j+1]
            if min_err > error:
                x = i
                y = j
                min_err = error
    if min_err > 500:
        xr = 0
        yr = 0
        print('Подозрение на ошибку в min_in_kvadrat не найден минимум')
    else:
        xr = x1 + x * shag1
        yr = y1 + y * shag2
        error = matrica[x,y]
    return xr, yr, error

# ищем для модели variant_model минимальной ошибки зависящей от 2-х параметров
def min_err_variant(raw, state):
    x1 = 50 # начало интервала 1-го параметра
    x2 = 4500 # конец интервала 1-го параметра
    y1 = -1.2 # начало интервала 2-го параметра
    y2 = 0.8 # конец интервала 2-го параметра
    err0 = model2_error(raw, state)
    delit_x = 4#10
    delit_y = 25
    shag_x = (x2 - x1) // delit_x
    shag_y = (y2 - y1) / delit_y
    x, y, err = min_in_kvadrat(x1, y1, shag_x, shag_y, delit_x, delit_y, raw, state, optim_variant)
    delit_x = 10
    delit_y = 10#20
    shag_x_1 = 2 * shag_x // delit_x
    shag_y_1 = 2 * shag_y / delit_y
    x, y, err = min_in_kvadrat(x - shag_x, y - shag_y, shag_x_1, shag_y_1, delit_x, delit_y, raw, state,
                               optim_variant)
    shag_x = shag_x_1
    shag_y = shag_y_1
    delit_x = 10
    delit_y = 10
    shag_x_1 = 2 * shag_x // delit_x
    shag_y_1 = 2 * shag_y / delit_y
    x, y, err = min_in_kvadrat(x - shag_x, y - shag_y, shag_x_1, shag_y_1, delit_x, delit_y, raw, state,
                               optim_variant)

    if err > err0:
        return 1, 1000, err0
    return x, y, err

# оптимизация применения variant_model в каждом штатае отдельно
def post_model_variant(raw, df_state):
    raw['y_prom'] = raw['y_inegr']
    for state in raw['state'].unique():
        rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
        param1, param2, error = min_err_variant(raw, state)
        rezult = model2_validacia(raw, rezult, state, param1, param2)
        # df_state.loc[len(df_state.index)] = [state, 0, 0, 0, param1, param2, error]
        df_state.loc[df_state['state'] == state, 'var_mod_1'] = param1
        df_state.loc[df_state['state'] == state, 'var_mod_2'] = param2
        df_state.loc[df_state['state'] == state, 'error'] = error
        print(df_state.head(60))

# выполнение оптимизированной var_model модели
def work_post_var_model(raw, row_state, maska):
    state = row_state['state']
    maska_state = maska & (raw['state'] == state)
    var_mod_1 = row_state['var_mod_1']
    var_mod_2 = row_state['var_mod_2']
    raw = state_post_var_model(raw, maska_state, var_mod_1, var_mod_2)
    return raw

# АЛГОРИТМ ПОИСКА ОПТИМАЛЬНОЙ ГРАНИЦЫ 'lastactive'. Используется в моделях trend_ok и granica

# 1 проход при поискe границы. Последним параметром передается функция выполнения модели
def serch_granica(raw, row_state, kol, param, max_gran, shag, serch_granica_for_model):
    minerr = 1000
    minparam = 150000
    state = row_state['state']
    while param <= max_gran:
        raw = serch_granica_for_model(raw, row_state, kol, param)
        error = model2_error(raw, state)
        if minerr > error:
            minerr = error
            minparam = param
        raw['y_inegr'] = raw['y_prom']
        param = round(param * shag + 1)
    return raw, minerr, minparam

# 2 прохода при поискe границы. Последним параметром передается функция выполнения модели
def oll_serch_granica(raw, row_state, kol, max_gran, serch_granica_for_model):
    global minimum
    param = minimum
    minparam = 0
    shag = 1.4
    raw, minerr, minparam = serch_granica(raw, row_state, kol, param, max_gran, shag, serch_granica_for_model)
    if minparam < 90000:
        param = minparam / shag
        max_gran = minparam * shag
        shag = 1.01
        raw, minerr, minparam = serch_granica(raw, row_state, kol, param, max_gran, shag, serch_granica_for_model)
    return raw, minerr, minparam

# МОДЕЛЬ MIX делающая разные смешения 'mbd_lag1' и 'y_inegr' для отдельных штатов и разных 'lastactive'
# выполнение модели mix с одним из 10 namb
def state_1_mix(raw, maska_state, granica, namb):
    maska_trend = maska_state &  (raw['lastactive'] < granica)
    namb = namb/10
    raw.loc[maska_trend, 'y_inegr'] = raw.loc[maska_trend, 'mbd_lag1']*(1-namb) + \
                                      raw.loc[maska_trend, 'ypred'] * namb
    return raw

# выполнение модели mix с параметрами namb от 9 до kol включительно
def state_kol_mix(raw, maska_state, row_state, kol):
    for namb in range(9, kol-1, -1):
        granica = row_state[f'g{namb}']
        raw = state_1_mix(raw, maska_state, granica, namb)
    return raw

# выполнение модели mix с параметрами namb от 9 до kol+1 включительно c c оптимизированными границами
# и с параметром kol с искомой granica
def state_mix(raw, row_state, kol, granica):
    global start_val, stop_model, minimum
    state = row_state['state']
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model) & (raw.lastactive > minimum)
    maska_state = maska & (raw['state'] == state)
    raw = state_kol_mix(raw, maska_state, row_state, kol+1)
    raw = state_1_mix(raw, maska_state, granica, kol)
    return raw

# оптимизация модели trend_ok для отдельного штата
def min_err_trend(raw, state):
    x1 = -0.1
    x2 = 1.1
    y1 = -0.1
    y2 = 1.1
    err0 = model2_error(raw, state)
    delit_x = 10
    delit_y = 10
    shag_x = (x2-x1)/delit_x+1
    shag_y = (y2-y1)/delit_y+1
    x, y, err = min_in_kvadrat(x1, y1, shag_x, shag_y, delit_x, delit_y, raw, state, optim_trend)
    delit_x = 10
    delit_y = 10
    shag_x_1 = 2 * shag_x / delit_x
    shag_y_1 = 2 * shag_y / delit_y
    x, y, err = min_in_kvadrat(x - shag_x, y - shag_y, shag_x_1, shag_y_1, delit_x, delit_y, raw, state,
                                optim_trend,)
    shag_x = shag_x_1
    shag_y = shag_y_1
    delit_x = 10
    delit_y = 10
    shag_x_1 = 2 * shag_x / delit_x
    shag_y_1 = 2 * shag_y / delit_y
    x, y, err = min_in_kvadrat(x - shag_x, y - shag_y, shag_x_1, shag_y_1, delit_x, delit_y, raw, state,
                                optim_trend,)
    if err >= err0:
        return -1, -1, err0
    return x, y, err

# оптимизация применения mix модели в каждом штатае отдельно
def post_model_mix(raw, df_state):
    global start_val, stop_model
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw['y_prom'] = raw['y_inegr']
    max_gran = 120000
    for kol in range(9,-1,-1):
        for i, row_state in df_state.iterrows():
            state = row_state['state']
            # raw['y_prom'] = raw['y_inegr']
            raw = work_post_var_model(raw, row_state, maska)
            rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
            #param1, param2, error = min_err_trend(raw, state)
            max_gran = 120000
            raw, error, param1 = oll_serch_granica(raw, row_state, kol, max_gran, state_mix)
            # max_gran = param1
            rezult = model2_validacia(raw, rezult, state, param1)
            # df_state.loc[len(df_state.index)] = [state, 0, param1, param2, 0, 0, error]
            df_state.loc[df_state['state'] == state, f'g{kol}'] = param1
            df_state.loc[df_state['state'] == state, 'error'] = error
            print(df_state.head(60))

# выполнение оптимизированной модели mix
def work_model_mix(raw, row_state, maska):
    state = row_state['state']
    maska_state = maska & (raw['state'] == state)
    state_kol_mix(raw, maska_state, row_state, 0)
    return raw

# ГРАНИЦА ПРАВИЛЬНОГО УЧЕТА У ШТАТОВ
# выполнение модели granica с заданной границей param
def state_post_granica(raw, maska_state, param):
    maska_gran = maska_state & (raw['lastactive'] < param)
    raw.loc[maska_gran, 'y_inegr'] = raw.loc[maska_gran, 'mbd_lag1']
    return raw

# выполнение модели granica с заданной границей param.
# Эта функция передается как параметр в serch_granica и oll_serch_granica
def state_post_model_granica(raw, row_state, param):
    global start_val, stop_model
    state = row_state['state']
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    maska_state = (raw.dcount >= start_val) & (raw.dcount <= stop_model) & (raw['state'] == state)
    raw = work_post_var_model(raw, row_state, maska)
    # raw = work_post_trend_ok(raw, row_state, maska)
    raw = state_post_granica(raw, maska_state, param)
    return raw

# поиск нижней границы 'lastactive' для каждого штата ниже которой предсказание некорректно
def granica_post_model(raw, df_state):
    raw['y_prom'] = raw['y_inegr']
    for i, row_state in df_state.iterrows():
        state = row_state['state']
        rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
        raw, minerr, minparam = oll_serch_granica(raw, row_state, state_post_model_granica)
        df_state.loc[df_state['state']==state, 'granica'] = minparam
        df_state.loc[df_state['state'] == state, 'error'] = minerr
        print(df_state.head(60))

# выполнение оптимизированной модели granica
def work_granica(raw, row_state, maska):
    state = row_state['state']
    maska_state = maska & (raw['state'] == state)
    granica = row_state['granica']
    raw = state_post_granica(raw, maska_state, granica)
    return raw

# ЗДЕСЬ ЗАКАНЧИВАЕТСЯ ОПТИМИЗАЦИЯ МОДЕЛЕЙ. Далее выполнение моделей.

# ЦФИПС МОДЕЛЬ
def cfips_1_mes(raw, mes_val, kol_mes):
    for cfips in raw['cfips'].unique():
        maska = (raw.cfips == cfips) & (raw.dcount < mes_val) & (raw.dcount >= mes_val-kol_mes)
        target = raw.loc[maska, 'microbusiness_density']
        ypred = raw.loc[maska, 'ypred']
        err_mod = smape(target, ypred)
        mbd_lag1 = raw.loc[maska, 'mbd_lag1']
        err_last = smape(target, mbd_lag1)
        maska = (raw.cfips == cfips) & (raw.dcount == mes_val)
        if err_mod > err_last:
            raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'mbd_lag1']
    return raw


def cfips_model(raw):
    global start_val, stop_model, main_start_val
    start_val = main_start_val
    for kol_mes in range(2,20): # количество месяцев учпитываемых в модели
        for mes_val in tqdm(range(main_start_val, stop_model + 1)):
            raw = cfips_1_mes(raw, mes_val, kol_mes)

            # mes_1 = 2  #
            # y_pred, znachenie = vsia_model(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0)
            # raw = post_model(raw, y_pred, mes_val, blac_test_cfips)
        rezult = model2_validacia(raw, rezult, kol_mes, 0, 0)
        rezult.sort_values(by='err_2mod', inplace=True, ascending=True)
        print(rezult.head(22))
        # otchet(raw, start_val, mes_val)


# выполнение можелей variant_model, trend_ok, granica
def work_state(raw, df_state):
    global start_val, stop_model
    rezult = pd.DataFrame(columns=['param1', 'param2', 'param3', 'err_2mod', 'dif_err_1mod', 'dif_err_last_mod'])
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_model)
    raw.loc[maska, 'y_inegr'] = raw.loc[maska, 'ypred']

    cfips_model(raw)


    # for i, row_state in df_state.iterrows():
    #     raw = work_post_var_model(raw, row_state, maska)
    # print('')
    # print('------оптимизированная по штатам variant_model ---------')
    # rezult = model2_validacia(raw, rezult)
    # model2_otchet(raw)
    # print(rezult.head(22))
    #
    # for i, row_state in df_state.iterrows():
    #     work_model_mix(raw, row_state, maska)
    # print('')
    # print('------оптимизированная по штатам trend_ok_model ---------')
    # rezult = model2_validacia(raw, rezult)
    # model2_otchet(raw)
    # print(rezult.head(22))
    # #
    # for i, row_state in df_state.iterrows():
    #     raw = work_granica(raw, row_state, maska)
    # print('')
    # print('------оптимизированная по штатам граница ---------')
    # rezult = model2_validacia(raw, rezult)
    # model2_otchet(raw)
    # print(rezult.head(22))

    # rezult = model2_validacia(raw, rezult)

# Оптимизация решения какую из моделей в каком случае применять на основе
def main_post_model(raw, N_modeli):
    raw['y_inegr'] = raw['ypred']

    # БЛОК ОПТИМИЗАЦИИ ПОСТ-МОДЕЛИ С ФОМИРОВАНИЕМ df_state
    # создание датафрейма в котором созраняем параметры оптимизации для штатов
    # sl ={'state':raw['state'].unique()}
    # df_state = pd.DataFrame(sl, columns=['state', 'g0', 'g1', 'g2', 'g3','g4','g5','g6','g7','g8','g9',
    #                                      'var_mod_1', 'var_mod_2','error'])

    # оптимизация для variant_model
    # df_state = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv")
    # post_model_variant(raw, df_state)
    # df_state.to_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv", index=False)

    #оптимизация для 'model_mix'
    # df_state = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv")
    # post_model_mix(raw, df_state)
    # df_state.to_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv", index=False)
    # #
    # оптимизация для нижней границы применимости первой модели
    # df_state = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv")
    # granica_post_model(raw, df_state)
    # df_state.to_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv", index=False)
    #
    # исполнение модели
    df_state = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\state{N_modeli}.csv")
    work_state(raw, df_state)

    # model2_otchet(raw)

def load_data():
    train, test, census, new_data = start()  # загрузка начальных файлов
    raw = maceraw(train, test)  # объединенный массив трейна и теста, создание объединенного raw
    raw = new_data_in_raw(raw, new_data)  # присоединение новых данных к raw
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw.csv", index=False)
    raw = censusdef(census, raw)  # присоединение файла census к raw
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw_cens.csv", index=False)
    return raw

# обработка января 2023 в связи с изменением численности населения по переписи
def january(raw, mnoshitel, colum): # colum имя колонки которую меняем
    raw = raw.join(mnoshitel, on='cfips')
    maska = (raw["year"]==2023)&(raw["month"]==1)
    raw.loc[maska, colum] = raw.loc[maska, colum] * raw.loc[maska, 'mnoshitel']
    raw.drop(columns = 'mnoshitel' , inplace = True)
    return raw

def otvet(raw):
    raw = january(raw, mnoshitel, 'ypred')
    test = raw[raw.istest == 1].copy()
    maska = test['microbusiness_density'].isnull()
    test.loc[maska,'microbusiness_density'] = test.loc[maska,'ypred']
    # test['microbusiness_density'] = test['ypred']
    test = test[['row_id', 'cfips', 'microbusiness_density']]
    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\main0.csv', index=False)

def model2_otvet(raw):
    raw = january(raw, mnoshitel, 'y_inegr')
    test = raw[raw.istest == 1].copy()
    maska = test['microbusiness_density'].isnull()
    test.loc[maska,'microbusiness_density'] = test.loc[maska,'y_inegr']
    test = test[['row_id', 'cfips', 'microbusiness_density']]
    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\model2.csv', index=False)

if __name__ == "__main__":
    N_modeli = 2
    minimum = 60 # ниже этого значения 'lastactive' считаем что 'microbusiness_density' не меняется
    # глобальные переменные

    stop_model = 41 # последний месяц до которого работает модель, включая месяцы теста
    stop_val = 40  # последний месяц валидации до которого проверяем модель
    main_start_val = 28 # первый месяц окончательной валидации с которого проверяем модель

    start_val = 6  # первый месяц предварительной валидации с которого проверяем модель
    pd.options.display.width = 0 # для печати

    # НАЧАЛЬНАЯ ЗАГРУЗКА
    #raw = load_data()
    # загрузка данных переписи
    mnoshitel = load_perepis()

    # датафрейм для хранения результатов вариантов оптимизации
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])

    # ЗАГРУЗКА ДАННЫХ
    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_cens.csv")
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0

    # ПЕРВЫЙ ПРОЦЕСС ОПТИМИЗАЦИИ
    segmenti = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\targ_diskretno{N_modeli}.csv")
    raw = model1_vse_fragmenti(raw, rezult, segmenti) # первая модель
    raw.to_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle1modeli{N_modeli}.csv", index=False)
    otvet(raw)

    start_val = main_start_val# 39

    # МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ какое предсказание лучше 1-й модели или модели не изменения 'microbusiness_density'
    # raw = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle1modeli{N_modeli}.csv")
    # raw['y_inegr'] = 0
    # variant_model(raw) #модель предсказывающая ошибку определения тренда первой моделью
    # raw.to_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle_2_model{N_modeli}.csv", index=False)

    # МОДЕЛЬ ПРЕДСКАЗЫВАЮЩАЯ правильно ли предсказан тренд в 1-й модели
    # raw = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle_2_model{N_modeli}.csv")
    # raw['y_inegr'] = 0
    # trend_ok_model(raw) #модель предсказывающая ошибку определения тренда первой моделью
    # raw.to_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle_2_model{N_modeli}.csv", index=False)


    # ОПРЕДЕЛЯЕМ В КАКОМ СЛУЧАЕ КАКУЮ МОДЕЛЬ ПРИМЕНЯТЬ

    raw = pd.read_csv(f"C:\\kaggle\\МикроБизнес\\raw_posle_2_model{N_modeli}.csv")
    main_post_model(raw, N_modeli) # выбираем первую модель или модель не изменения 'microbusiness_density'

    # СОЗДАЕМ ФАЙЛ С РЕЗУЛЬТАТОМ
    model2_otvet(raw)
