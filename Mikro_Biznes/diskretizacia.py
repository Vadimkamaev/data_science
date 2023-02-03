# разделеение на части по рамеру 'lastactive'
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm # прогресс бар
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from servise_ds import okno
import obrabotka_filtr

d = {'granica': [80, 250, 600, 1300, 3200, 4500, 1000000000000],
     'lastactive': [20,0,0,0,0,0,0],
     'param':[20,0,0,0,0,0,0]}
sedmenti = pd.DataFrame(d)

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

# УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
def del_outliers(raw, l_vibr=0.018, verh=0.0144, niz = 0.0046): # l_vibr, verh, niz - параметры для оптимизации
    # для verh и niz 0.0045 лучше чем 0.003 и чем 0.006
    raw['mbd_gladkaya']=0 # в результате функции создается новая колонка - сглаженный 'microbusiness_density'
    raw['mbd_gladkaya_dif'] = 0  # изменение сглаженного 'microbusiness_density'
    outliers = []  # список выбросов
    cnt = 0  # счетчик выбросов
    vsego = 0
    # цикл по уникальным cfips
    print('Удаление выбросов')
    for cfips in tqdm(raw.cfips.unique()):  # tqdm - прогресс бар
        maska_cfips = (raw['cfips'] == cfips)  # маска отфильтрованная по cfips
        tmp = raw.loc[maska_cfips].copy().reset_index(drop=True)  # df фильтрован по cfips
        # массив значений microbusiness_density при cfips из цикла
        mb_d = tmp.microbusiness_density.values.copy()
        # цикл назад от предпоследнего до 2-го элемента
        for i in range(38, 2, -1): # у умного чувака было с 37 по 2(не вкл.) элемента вниз
            # среднее значение microbusiness_density с 0-го по i-й элемент * l_vibr
            thr = l_vibr * np.mean(mb_d[:i])
            difa = mb_d[i] - mb_d[i - 1]
            if (difa >= thr) or (difa <= -thr):  # если microbusiness_density изменился больше чем на l_vibr%
                if difa > 0:
                    mb_d[:i] += difa - verh # verh=0.0144 error 1.791025, dif_err -0.401444
                else:
                    mb_d[:i] += difa + niz # 0.0046 error 1.791025, dif_err -0.401444
                outliers.append(cfips)  # добавляем cfips в список выбросов
                cnt += 1  # счетчик выбросов
            vsego += 1
        # mb_d[0] = mb_d[1] * 0.99  # у умного чувака была эта фигня
        raw.loc[maska_cfips, 'mbd_gladkaya'] = mb_d
        raw.loc[maska_cfips, 'mbd_gladkaya_dif'] = raw.loc[maska_cfips, 'mbd_gladkaya'].diff(1).fillna(0)
    outliers = np.unique(outliers)
    print('Кол. cfips с разрывами =', len(outliers), 'Кол. разрывов =', cnt, '% разрывов',cnt/vsego*100)
    return raw

# УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
def del_outliers_max(raw, max_cfips, l_vibr=0.018, verh=0.0144, niz = 0.0046): # l_vibr, verh, niz - параметры для оптимизации
    # для verh и niz 0.0045 лучше чем 0.003 и чем 0.006
    # raw['mbd_gladkaya']=0 # в результате функции создается новая колонка - сглаженный 'microbusiness_density'
    # raw['mbd_gladkaya_dif'] = 0  # изменение сглаженного 'microbusiness_density'
    outliers = []  # список выбросов
    cnt = 0  # счетчик выбросов
    vsego = 0
    # цикл по уникальным cfips
    print('Удаление выбросов')
    df = raw[raw['cfips']>max_cfips]
    for cfips in tqdm(df.cfips.unique()):  # tqdm - прогресс бар
        maska_cfips = (raw['cfips'] == cfips)  # маска отфильтрованная по cfips
        tmp = raw.loc[maska_cfips].copy().reset_index(drop=True)  # df фильтрован по cfips
        # массив значений microbusiness_density при cfips из цикла
        mb_d = tmp.microbusiness_density.values.copy()
        # цикл назад от предпоследнего до 2-го элемента
        for i in range(38, 2, -1): # у умного чувака было с 37 по 2(не вкл.) элемента вниз
            # среднее значение microbusiness_density с 0-го по i-й элемент * l_vibr
            thr = l_vibr * np.mean(mb_d[:i])
            difa = mb_d[i] - mb_d[i - 1]
            if (difa >= thr) or (difa <= -thr):  # если microbusiness_density изменился больше чем на l_vibr%
                if difa > 0:
                    mb_d[:i] += difa - verh # verh=0.0144 error 1.791025, dif_err -0.401444
                else:
                    mb_d[:i] += difa + niz # 0.0046 error 1.791025, dif_err -0.401444
                outliers.append(cfips)  # добавляем cfips в список выбросов
                cnt += 1  # счетчик выбросов
            vsego += 1
        # mb_d[0] = mb_d[1] * 0.99  # у умного чувака была эта фигня
        raw.loc[maska_cfips, 'mbd_gladkaya'] = mb_d
        raw.loc[maska_cfips, 'mbd_gladkaya_dif'] = raw.loc[maska_cfips, 'mbd_gladkaya'].diff(1).fillna(0)
    outliers = np.unique(outliers)
    if vsego != 0:
        print('Кол. cfips с разрывами =', len(outliers), 'Кол. разрывов =', cnt, '% разрывов',cnt/vsego*100)
    return raw

# создание лагов
def build_lag(raw):
    # создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    train_col = ['mbd_lag1']  # список полей используемых в трайне
    # создаем лаги 'microbusiness_density' с учетом сглаживания от 1 до lags
    for lag in range(1, 3):
        # shift - сдвиг на определеное кол. позиций
        raw[f'mbd_lag{lag+1}'] = raw[f'mbd_lag{lag}'] - raw.groupby('cfips')['mbd_gladkaya_dif'].shift(lag)
        train_col.append(f'mbd_lag{lag + 1}')
        raw[f'mbd_lag{lag+1}'].fillna(method='bfill', inplace=True)
    li = [3 ,7, 10]
    # создаем скользящие средние.
    for i in li:
        nam = f'EMA_{i}'  # лучшая скользящая средняя - 3 : error 1.791025, dif_err -0.401444
        EMA = pd.Series(raw['mbd_lag1'].ewm(span=i, adjust=False, min_periods=1).mean(), name=nam)
        raw[nam] = EMA
        train_col += [nam]
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['active_lag1']
  #  df = raw[['row_id', 'microbusiness_density', 'mbd_gladkaya', 'mbd_gladkaya_dif', nam]+train_col]
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'microbusiness_density']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'microbusiness_density']
    return X_test, y_test

# Сохраняем результат.
def sbor_rezult(raw, y_pred, y_test, mes_val):
    #vsmape(y_test, y_pred)
    #Почему то у умного чувака mes_val + 1 ! Вроде должно быть mes_val
    raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred
    return raw

# создание блек листа 'cfips' для теста, может делать это моделью
def mace_blac_list(raw, mes_1, mes_val, blac_cfips, param=-1):
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    raw['error_otdelno'] = vsmape(raw['microbusiness_density'], raw['ypred_otdelno'])
    # создаем датафрейм со столбцами error и error_last и индексом 'cfips' + 'dcount'
    dt = raw.loc[(raw.dcount >= mes_1) & (raw.dcount <= mes_val)].groupby(['cfips', 'dcount'])[['error', 'error_last','lastactive']].last()
    # добавляем в dt столбец булевых значений 'miss' - количество ошибок > или <
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    seria_dt = dt.groupby('cfips')['miss'].mean()
    ser_err = dt.groupby('cfips')['error'].mean()
    ser_lastactive = dt.groupby('cfips')['lastactive'].mean()
    ser_error_last = dt.groupby('cfips')['error_last'].mean()
    df_dt = pd.DataFrame({'cfips': seria_dt.index, 'lastactive':ser_lastactive, 'miss': seria_dt, 'dif_err':ser_error_last-ser_err})
    seria_dt = seria_dt.loc[seria_dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(seria_dt))
    # Признак отправления в блек лист
    #blec_maska = df_dt['dif_err'] < param # param от 0 (большой блек) до -10 (малый блек). Проблема 1 выброс
    blec_maska = (df_dt['miss'] >= 0.5) # от 0.5 (большой блек) до 1 (малый блек) Проблема на 1-м месяце исключит всех
    #df_dt.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)
    #df_dt = df_dt[~blec_maska]
    df_dt.sort_values(by='lastactive', inplace=True)
    shad = 10
    for lastactive in range(50, 250, shad):
        maska = (df_dt['lastactive']>lastactive)&(df_dt['lastactive']<=lastactive+shad)
        horoshie = df_dt[~blec_maska&maska].count()['lastactive']
        vsego = df_dt[maska].count()['lastactive']
        print(f'От {lastactive} до {lastactive+shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie/vsego*100}')
    lastactive = lastactive+shad
    shad = 1000000000
    maska = (df_dt['lastactive'] > lastactive) & (df_dt['lastactive'] <= lastactive + shad)
    horoshie = df_dt[~blec_maska & maska].count()['lastactive']
    vsego = df_dt[maska].count()['lastactive']
    print(f'От {lastactive} до {lastactive + shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie / vsego * 100}')

    pass
    # blac_cf = list(blac_cf)
    # #if mes_val > mes_1+5: 1.524041 -0.233395     0.041235
    # if mes_val > mes_1 + 3:
    #     print('Размер блек листа', len(blac_cf))
    #     blac_cfips = list(blac_cfips)+blac_cf
    return blac_cfips

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
def validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    # maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
    #         (raw.cfips <= max_cfips) & (raw.lastactive > lastactive)  & (~raw['cfips'].isin(blac_cfips))
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & \
            (raw.cfips <= max_cfips)  & (~raw['cfips'].isin(blac_test_cfips))
    target = raw.loc[maska, 'microbusiness_density']
    ypred = raw.loc[maska, 'ypred']
    err_mod = smape(target, ypred)
    print('Предсказано иссдедуемой моделью. Ошибка SMAPE:',err_mod)
    mbd_lag1 = raw.loc[maska, 'mbd_lag1']
    err_last = smape(target, mbd_lag1)
    print('Равенство последнему значению. Ошибка SMAPE:', err_last)

    y_no_blac = raw.loc[maska, 'y_no_blac']
    err_y_no_blac = smape(target, y_no_blac)
    print('Без блек листа. Ошибка SMAPE:', err_y_no_blac)

    # смотрел зависимость ошибок от ластактиве. Первое успешное предсказание при lastactive = 10
    # raw['error'] = vsmape(raw['microbusiness_density'], raw['y_no_blac'])
    # raw['error_last'] = vsmape(raw['microbusiness_density'], raw['mbd_lag1'])
    # raw['diferror'] = raw['error_last']-raw['error']
    # raw.sort_values(by=['lastactive','row_id'],inplace=True)
    # raw = raw[(raw.dcount >= start_val) & (raw.dcount <= stop_val) & (raw.cfips <= max_cfips)]
    # df = raw[['cfips','lastactive','diferror']].groupby(['lastactive','cfips']).mean()
    # print('lastactive min=', raw[(raw['error']>0)&(raw.dcount >= start_val) & (raw.dcount <= stop_val) &
    #                              (raw.cfips <= max_cfips)]['lastactive'].min())

    #print_state(raw, maska) # вывод на печать информацию по качеству работы модели в каждом штате
    #print_month(raw, maska)

    dif_err = err_last - err_mod # положительная - хорошо
    dif_no_blac = err_y_no_blac - err_mod # положительная - хорошо
    rezult.loc[len(rezult.index)] = [param1, param2, param3, err_mod, dif_err, dif_no_blac]
    return rezult

def vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0):
    # получение трайна и 'y' (игрик) для модели
    df = raw[~raw['cfips'].isin(blac_cfips)]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    # Создаем модель
    #model = RandomForestRegressor(n_estimators=100, min_samples_split=50, n_jobs=2, random_state=322) #error 7.977319
    # лучшее n_estimators=900 лучше чем 500, лучше чем 100,  разница ошибки между 900 и 100 - 0.5%
    # 100 деревьев в 5 раз быстрее 900 деревьев
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=1000,
        learning_rate=0.0108, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)
    # сохраняем результат обработки одного цикла
    raw.loc[raw.dcount == mes_val, 'ypred'] = y_pred
    # прибавляем значимость столбцов новой модели к значениям предыдущих
    znachenie['importance'] = znachenie['importance'] + model.feature_importances_.ravel()
    return raw, znachenie

def modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips):
    start_time = datetime.now()  # время начала работы модели
    znachenie = pd.DataFrame({'columns': train_col, 'importance': 0})
    # c start_val начинаeтся цикл по перебору номера валидационного месяца до stop_val включительно
    for mes_val in tqdm(range(start_val, stop_val + 1)):  # всего 39 месяцев с 0 до 38 в трайне заполнены инфой
        # здесь должен начинаться цикл по перебору номера первого месяца для трайна
        mes_1 = 4 # error 1.598877, dif_err -0.287906, длинная модель
        raw, znachenie = vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0)
    #blac_cfips = mace_blac_list(raw, start_val, mes_val, blac_cfips)
    print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
    # print('Значимость колонок трайна в модели')
    # znachenie.sort_values(by='importance', ascending=False, inplace=True)
    # print(znachenie)
    return raw, blac_cfips

# создание новых фиктивных cfips
def new_cfips(raw, lastactive, max_cfips, kol = 2):
    #kol - оптимально 2
    if max_cfips != raw['cfips'].max():
        print('Ошибка max_cfips')
    df = raw[raw['lastactive'] < lastactive].copy()
    df.sort_values('lastactive', ascending=False, inplace=True)
    if len(df.index) == 0:
        return raw
    list_cfips = df['cfips'].unique() #  массив уникальных cfips, которые группируются
    lec = len(list_cfips) # количество уникальных cfips, которые группируются
    le = lec + kol - lec%kol # длина матрицы - количество новых генерируемых cfips
    matrica = np.zeros((kol,le)) # матрица каждая строка которой - список cfips которые суммируются
    fin = le // kol  # номер элемента до которого все cfips использованы в 0-м ряду и больше их не применять
    for l in range(fin): # цикл по заполнению 0-го ряда
        for k in range(kol):
            x = list_cfips[l]
            matrica[0,l*kol+k] = x
    sled = fin # номер следующего cfips который вставляется в матрицу
    for k in range(1, kol): # цикл по заполнению рядов с 1 по последний
        for l in range(le-1,-1,-1): # цикл в котором заполняется один ряд матрицы
            if list_cfips[sled] == matrica[k-1, l]:
                sled+=1
            matrica[k, l] = list_cfips[sled]
            if sled == lec-1:
                sled = fin
            else:
                sled+=1
    # матрица заполнена, далее суммирование cfips
    # считываем стандартные колонки одинаковые у всех cfips
    cfips = list_cfips[0]
    dfcop = raw[raw['cfips']==cfips].copy()
    dfcop.reset_index(drop=True, inplace=True)
    dfcop[['row_id', 'cfips', 'county','state', 'first_day_of_month',
           'microbusiness_density', 'active',
           'county_i', 'pct_bb', 'pct_college', 'pct_foreign_born',
           'pct_it_workers', 'median_hh_inc', 'covill', 'covdeat', 'Population',
           'proc_covill', 'proc_covdeat', 'ypred',
           'error_otdelno', 'mbd_gladkaya', 'mbd_gladkaya_dif', 'mbd_lag1',
           'mbd_lag2', 'mbd_lag3', 'EMA_3', 'active_lag1', 'ypred_otdelno',
           'lastactive']] = 0
    dfcop['state_i']=100
    for l in range(le):
        nas = 0
        gen_cfips = max_cfips+l+1 # номер нового сгенерированного cfips
        dfnew = dfcop.copy()
        dfnew['cfips'] = gen_cfips
        df =kol*[0]
        for i in range(kol):
            df[i] = raw[raw['cfips'] == matrica[i, l]].copy()
            df[i].reset_index(drop=True, inplace=True)
            dfnew['active'] = dfnew['active'] + df[i]['active']
            dfnew['lastactive'] = dfnew['lastactive'] + df[i]['lastactive']
            nas = nas + df[i]['Population'][0]
            dfnew['proc_covill'] = dfnew['proc_covill'] + df[i]['proc_covill']*df[i]['Population'][0]
            dfnew['pct_college'] = dfnew['pct_college'] + df[i]['pct_college'] * df[i]['Population'][0]
            dfnew['median_hh_inc'] = dfnew['median_hh_inc'] + df[i]['median_hh_inc'] * df[i]['Population'][0]
        dfnew['microbusiness_density'] = dfnew['active']/nas
        dfnew['proc_covill'] = dfnew['proc_covill']/nas
        dfnew['pct_college'] = dfnew['pct_college']/nas
        dfnew['median_hh_inc'] = dfnew['median_hh_inc'] /nas
        raw = pd.concat([raw,dfnew], ignore_index=True)
    return raw

# после сглаживания
def posle_sglashivfnia(raw, rezult):
    # 'lastactive' содержит значение 'active' одинаковое для всех строк с одним 'cfips'
    # и равное 'active' за последний месяц (сейчас 2022-10-01)
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')

    # создаем блек-лист cfips которых не используем в тесте
    blac_test_cfips = raw.loc[(raw['lastactive'] < 600)|(raw['lastactive'] >= 1300), 'cfips']
    #blac_test_cfips = raw.loc[(raw['lastactive'] < 50), 'cfips']
    blac_test_cfips = blac_test_cfips.unique()
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    for kol in [2]:
        for max_active in [1000000000000,5500,15200,20250] + list(range(16000,30000,1000)): # 1.110824 -0.061638     0.051209
            for param in range(0,1,5):
                #max_active = 20250000000000000000
                param = max_active
                lastactive = 70
                raw = raw[raw['cfips'] <= max_cfips]
                # создаем блек-лист cfips который не нужен в трайне  1.997686 -0.152959     0.082025
                maska = (raw['lastactive']>max_active)|(raw['lastactive']<=lastactive)
                blac_cfips = raw.loc[maska, 'cfips'] # оптимально с lastactive до 32 убираем из трайна
                blac_cfips = blac_cfips.unique()
                # создаем гибриды из cfips меньше 29 (оптимально) для трайна
                raw = new_cfips(raw, lastactive, max_cfips)
                # сглаживаем гибриды
                #del_outliers_max(raw, max_cfips, l_vibr=0.018, verh=0.0144, niz = 0.0046)

                # здесь должен начинаться цикл по количеству лагов
                raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
                # здесь должен начинаться цикл перебирающий все комбинации из списка полей
                train_col += ['state_i', 'proc_covill', 'pct_college', 'median_hh_inc', 'sp500', 'month']

                # возможные поля 'Population', 'proc_covdeat', 'pct_bb', 'pct_foreign_born', 'pct_it_workers', 'unemploy'
                start_val = 30  # первый месяц валидации с которого проверяем модель
                stop_val = 38  # последний месяц валидации до которого проверяем модель
                # валидация по результату обработки всех месяцев валидации в цикле
                raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips)
                rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, lastactive, param, kol)
        # здесь заканчиваются циклы оптимизации
        #rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim1.csv", index=False)
                print('Сортировка по dif_no_blac')
                rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
                print(rezult.head(22))
                print('Сортировка по dif_err')
                rezult.sort_values(by='dif_err', inplace=True, ascending=False)
                print(rezult.head(22))

# определяем ошибку модели для сегмента cfips с lastactive в интервале от mini до maxi
# при параметрах min_type=0, min_blac=-1, max_blac=0 определяющих блек лист
def serch_error(raw, rezult, mini, maxi, min_type=0, min_blac=-1, max_blac=0):
    # создаем блек-лист cfips которых не используем в тесте
    maska = (raw['lastactive'] < mini)
    if maxi > 1:
        maska = maska|(raw['lastactive'] >= maxi)
    blac_test_cfips = raw.loc[maska, 'cfips']
    blac_test_cfips = blac_test_cfips.unique()
    # создаем блек-лист cfips который не нужен в трайне
    maska = (raw['lastactive'] <= min_blac)
    if max_blac > 1:
        maska = (raw['lastactive'] > max_blac) | maska
    blac_cfips = raw.loc[maska, 'cfips']  #
    blac_cfips = blac_cfips.unique()
    max_cfips = raw['cfips'].max()
    # создаем гибриды из cfips меньше 29 (оптимально) для трайна
    if min_type > 0.5: # если min_type = 1 создаем гибриды, если = 0 то удаляем из трайна
        raw = new_cfips(raw, min_blac, max_cfips)
    # сглаживаем гибриды
    # del_outliers_max(raw, max_cfips, l_vibr=0.018, verh=0.0144, niz = 0.0046)

    # здесь должен начинаться цикл по количеству лагов
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
    # здесь должен начинаться цикл перебирающий все комбинации из списка полей
    train_col += ['state_i', 'proc_covill', 'pct_college', 'median_hh_inc', 'sp500', 'month']

    # возможные поля 'Population', 'proc_covdeat', 'pct_bb', 'pct_foreign_born', 'pct_it_workers', 'unemploy'
    start_val = 38#30  # первый месяц валидации с которого проверяем модель
    stop_val = 38  # последний месяц валидации до которого проверяем модель
    # запускаем модель по всем месяцам
    raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips)
    # валидация по результату обработки всех месяцев валидации в цикле
    rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, max_cfips, min_type, min_blac, max_blac)
    # здесь заканчиваются циклы оптимизации
    error = rezult.loc[len(rezult.index)-1,'error']
    return error, rezult


# оптимизируем сегмент cfips в интервале от mini до maxi
# составляем список из 5 лучших возможных max_active - в трайне cfips с lastactive < max_active
def serch_max_active(raw, mini, maxi, rezult):
    max_cfips = raw['cfips'].max()  # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    # задаем начальный шаг поиска оптимальной верхней границы lastactive
    # до которой мы используем cfips в трайне
    #shag = maxi + 300
    potolok = maxi*20 +20000
    max_active = maxi*10 # начальное значение с которого начинаем поиск
    # определяем ошибку модели
    error, rezult = serch_error(raw, rezult, mini, maxi, min_type=0, min_blac=-1, max_blac=0)
    # создаем датафрейм для хранения лучших результатов
    dfmax_active = pd.DataFrame({'max_active':[max_active], 'error':[error]})
    # первичный проход с поиском лучших результатов - минимальных ошибок
    while max_active < potolok:
        max_active = round(max_active *1.1)
        raw = raw[raw['cfips'] <= max_cfips]
        # определяем ошибку модели
        error, rezult = serch_error(raw, rezult, mini, maxi, min_type=0, min_blac=-1, max_blac=max_active)
        dfmax_active.loc[len(dfmax_active.index)]=[max_active,error]

    # rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim1.csv", index=False)
    rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(50))
    del_shag = 2 # делитель шага; во сколько раз уменьшается шаг на каждом следующем уровне
    # множество в котором количество строк в raw соответствующее проверенным значений max_active
    ispolzovano  = set()
    while del_shag < 100:  # цикл с уменьшением шага
        dfmax_active.sort_values(by='error', inplace=True, ignore_index=True)
        dfmax_active = dfmax_active.loc[0:4] # оставляем 5 лучших вариантов
        for ind in range(5): # проверяем значения близкие к лучшим
            row = dfmax_active.loc[ind] # строка датафрейма
            mini_shag = round(row['max_active'] * 0.1 / del_shag)
            if mini_shag >= 1: # Проверить коэффициент
                max_active = row['max_active'] + mini_shag # проверяем лучшее значение + пол шага
                cfips_in = (raw['lastactive'] > max_active).sum()
                if not (cfips_in in ispolzovano): # если такого значения max_active еще не было
                    ispolzovano.add(cfips_in)
                    raw = raw[raw['cfips'] <= max_cfips]
                    # определяем ошибку модели
                    error, rezult = serch_error(raw, rezult, mini, maxi, min_type=0, min_blac=-1, max_blac=max_active)
                    dfmax_active.loc[len(dfmax_active.index)] = [max_active, error]
                max_active = row['max_active'] - mini_shag # проверяем лучшее значение - пол шага
                cfips_in = (raw['lastactive'] > max_active).sum()
                if not (cfips_in in ispolzovano): # если такого значения max_active еще не было
                    ispolzovano.add(cfips_in)
                    raw = raw[raw['cfips'] <= max_cfips]
                    # определяем ошибку модели
                    error, rezult = serch_error(raw, rezult, mini, maxi, min_type=0, min_blac=-1, max_blac=max_active)
                    dfmax_active.loc[len(dfmax_active.index)] = [max_active, error]
        del_shag = del_shag * 2
    rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(50))
    dfmax_active.sort_values(by='error', inplace=True, ignore_index=True)
    dfmax_active = dfmax_active.loc[0:4] # оставляем 5 лучших вариантов
    print('Сортировка по dif_no_blac')
    rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(50))
    return dfmax_active, rezult # возвращаем 5 лучших вариантов

def serch_min_active(raw, max_active, mini, maxi, rezult):
    pass

# оптимизируем один сегмент cfips с lastactive в интервале от mini до maxi
def optimizacia_segmenta(raw, mini, maxi, rezult):
    if maxi > 1:
        df_active, rezult = serch_max_active(raw, mini, maxi, rezult)
    else:
        df_active = pd.DataFrame({'max_active':[0], 'error':[0]})
    df_active['min_active'] = 0
    df_active['min_type'] = 0
    for ind, row in df_active.iterrows():
        df_active, rezult = serch_min_active(raw, row['max_active'], mini, maxi, rezult)
    df_active.sort_values(by='error', inplace=True, ignore_index=True)
    max_active = df_active.loc[0,'max_active']
    min_active = df_active.loc[0, 'min_active']
    min_type = df_active.loc[0, 'min_type']
    error, rezult = serch_error(raw, rezult, mini, maxi, min_type, min_blac=min_active, max_blac=max_active)
    error = rezult.loc[len(rezult.index) - 1, 'error']
    dif_err = rezult.loc[len(rezult.index) - 1, 'dif_err']
    dif_no_blac = rezult.loc[len(rezult.index) - 1, 'dif_no_blac']
    return min_type, min_active, max_active, error, dif_err, dif_no_blac


# распечатываем датафрейм sedmenti
def print_sedmenti(sedmenti):
    for versia in sedmenti['versia'].unique():
        maska = sedmenti['versia'] == versia
        df = sedmenti[maska]
        print(f"Версия = {versia}. Оптимизированно на {df['optim'].mean()*100}%")
        print(f"'error':{df['error'].mean()}; 'dif_err':{df['dif_err'].mean()}; "
              f"'dif_no_blac':{df['dif_no_blac'].mean()}")
        print(df.head(40))

# оптимизируем все сегменты определенной версии в датафрейме sedmenti
def optimizacia_segmentacii(raw, rezult):
    versia = 1
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    sedmenti = pd.read_csv("C:\\kaggle\\МикроБизнес\\sedmenti.csv")
    df = sedmenti[(sedmenti['versia'] == versia)&(sedmenti['optim'] == 0)]
    for ind, row in df.iterrows():
        min_type, min_active, max_active, error, dif_err, dif_no_blac = \
            optimizacia_segmenta(raw, row['min'], row['max'], rezult)
        sedmenti.loc[ind,'min_type'] = min_type
        sedmenti.loc[ind, 'min_active'] = min_active
        sedmenti.loc[ind, 'max_active'] = max_active
        sedmenti.loc[ind, 'error'] = error
        sedmenti.loc[ind, 'dif_err'] = dif_err
        sedmenti.loc[ind, 'dif_no_blac'] = dif_no_blac
        sedmenti.loc[ind, 'optim'] = 1
        #sedmenti.to_csv("C:\\kaggle\\МикроБизнес\\sedmenti.csv", index=False)
    print_sedmenti(sedmenti)

# подсчет количества элементов в каждом сегменте и запись его в датафрейм sedmenti
def kol_in_sedmenti(raw, sedmenti, versia):
    maska = sedmenti['versia'] == versia
    df = sedmenti[maska]
    for ind, row in df.iterrows():
        maska = raw['lastactive'] >= row['min']
        if row['max'] !=0:
            maska = maska & (raw['lastactive'] < row['max'])
        kol = raw[maska]['cfips'].unique()
        kol = len(kol)
        sedmenti.loc[ind,'kol'] = kol
    return sedmenti

# создание датафрейма sedmenti и сохранение его в файл
def init_segmentacii(raw):
    granici =[50, 100, 150, 220, 330, 500, 800, 1200, 2200, 5000, 15000]
    versia = 1
    d = {'versia':versia,  #возможно несколько версий разбиения в процессе оптимизации
         'min': granici, # 'lastactive' >= 'min'
         'max': granici[1:]+[0], # 'lastactive' < 'max'
         'kol': 0, # количество cfips в интервале
         # параметры оптимизации
         'min_type': 0, # 0,1; 0 - удаление значений меньше 'min_active'; 1 - их группировка
         'min_active': 0, # cfips c 'lastactive' <= 'min_active' удаляются из трайна или группируются
         'max_active': 0, # cfips c 'lastactive' > 'max_active' удаляются из трайна
         # результат оптимизации
         'optim':0, # 0 - этот интервал от 'min' до 'max' не оптимизирован, 1 - оптимизирован
         'error':0,
         'dif_err':0,
         'dif_no_blac':0}
    sedmenti = pd.DataFrame(d)
    sedmenti = kol_in_sedmenti(raw, sedmenti, versia)
    print_sedmenti(sedmenti)
    #sedmenti.to_csv("C:\\kaggle\\МикроБизнес\\sedmenti.csv", index=False)

# создание raw с предсказаниями без блек листов с использованием в трайне всех cfips
def no_blec_vse_cfips(raw, rezult):
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906
    train_col += ['state_i', 'proc_covill', 'pct_college', 'median_hh_inc', 'sp500', 'month']
    start_val = 30  # первый месяц валидации с которого проверяем модель
    stop_val = 38  # последний месяц валидации до которого проверяем модель
    # модель без блек листа
    raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, [])
    raw['y_no_blac'] = raw['ypred']
    raw.to_csv("C:\\kaggle\\МикроБизнес\\rawy_no_blac.csv", index=False)

    # валидация по результату обработки всех месяцев валидации в цикле
    raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips)
    rezult = validacia(raw, start_val, stop_val, rezult, [], 10000000, 1, 1, 1)
    #rezult.to_csv("C:\\kaggle\\МикроБизнес\\rez_optim1.csv", index=False)
    rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(22))

# МОДЕЛЬ.
def glavnaia(raw, rezult):
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['median_hh_inc'] = raw.groupby('cfips')['median_hh_inc'].ffill()
    raw['ypred'] = 0
    max_cfips = raw['cfips'].max() # максимальная реальна 'cfips', больше неё фиктивные 'cfips'
    raw = del_outliers(raw, l_vibr=0.018, verh=0.0144, niz=0.0046)  # сглаживание, убираем выбросы
    raw = raw[raw['cfips'] <= max_cfips]
    posle_sglashivfnia(raw, rezult)

if __name__ == "__main__":
    pd.options.display.width = 0
    # train, test = start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")
    # glavnaia(raw, rezult)
    #raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_otdelno.csv")

    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\rawy_no_blac.csv")
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0
    #init_segmentacii(raw)
    optimizacia_segmentacii(raw, rezult)
    #glavnaia(raw, rezult)
   # posle_sglashivfnia(raw, rezult)