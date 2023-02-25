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
    return raw

# создание лагов  error 1.379034  0.010546  146.796656
def build_lag(raw): #
    train_col = []  # список полей используемых в трайне
    #создаем лаги 'target'
    for lag in range(1, 17): #17 -  1.409514  0.009615   144.354724
        raw[f'target_lag{lag}'] = raw.groupby('cfips')['target'].shift(lag)
        train_col.append(f'target_lag{lag}')
    #создаем 1 лаг 'microbusiness_density'
    raw['mbd_lag1'] = raw.groupby('cfips')['microbusiness_density'].shift(1)
    raw['mbd_lag1'].fillna(method='bfill', inplace=True)
    # создаем 1 лаг 'active' - общее количество микропредприятий в округе
    raw['active_lag1'] = raw.groupby('cfips')['active'].shift(1)
    train_col += ['mbd_lag1', 'active_lag1', 'median_hh_inc', 'month'] #  1.403340  0.015789   144.360897
    return raw, train_col

# получение трайна и 'y' (игрик) для модели, mes_1 - первый месяц с которого используем трайн для модели
def train_and_y(raw, mes_1, mes_val, train_col): # train_col - список используемых полей трейна
    # маска тренировочной выборки 1.408914 -0.097942
    maska_train = (raw.istest == 0) & (raw.dcount < mes_val) & (raw.dcount >= mes_1)
    train = raw.loc[maska_train, train_col]
    y = raw.loc[maska_train, 'target']
    return train, y

# Получение х_тест и y_тест. mes_val - месяц по которому проверяем модель
def x_and_y_test(raw, mes_val, train_col): # train_col - список используемых полей трейна
    # маска валидационной выборки. Валидация по 1 месяцу
    maska_val = (raw.istest == 0) & (raw.dcount == mes_val)
    X_test = raw.loc[maska_val, train_col]
    y_test = raw.loc[maska_val, 'target']
    return X_test, y_test

# создание блек листа 'cfips' для теста, может делать это моделью
def mace_blac_list(raw, mes_1, mes_val):
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
    # blec_maska = (df_dt['miss'] >= 0.5) # от 0.5 (большой блек) до 1 (малый блек) Проблема на 1-м месяце исключит всех
    #
    # df_dt.sort_values(by='lastactive', inplace=True)
    # shad = 10
    # for lastactive in range(50, 250, shad):
    #     maska = (df_dt['lastactive']>lastactive)&(df_dt['lastactive']<=lastactive+shad)
    #     horoshie = df_dt[~blec_maska&maska].count()['lastactive']
    #     vsego = df_dt[maska].count()['lastactive']
    #     print(f'От {lastactive} до {lastactive+shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie/vsego*100}')
    # lastactive = lastactive+shad
    # shad = 1000000000
    # maska = (df_dt['lastactive'] > lastactive) & (df_dt['lastactive'] <= lastactive + shad)
    # horoshie = df_dt[~blec_maska & maska].count()['lastactive']
    # vsego = df_dt[maska].count()['lastactive']
    # print(f'От {lastactive} до {lastactive + shad} :хор.{horoshie}, всего {vsego}, % хор={horoshie / vsego * 100}')
    return

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
    granica = 0.2
    target = raw.loc[maska, 'microbusiness_density']
    m = raw['microbusiness_density'] > (1 + granica) * raw['mbd_lag1']
    target.loc[m] = raw.loc[maska & m, 'mbd_lag1'] * (1 + granica)
    m = raw['microbusiness_density'] < (1 - granica) * raw['mbd_lag1']
    target.loc[m] = raw.loc[maska & m, 'mbd_lag1'] * (1 - granica)
    return target

# Валидация
def validacia(raw, start_val, stop_val, rezult, blac_test_cfips, param1=1, param2=1, param3=1):
    # маска при которой была валидация и по которой сверяем ошибки
    maska = (raw.dcount >= start_val) & (raw.dcount <= stop_val) & (~raw['cfips'].isin(blac_test_cfips))
    target = target_for_error(raw, maska)
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
    return rezult

def vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0):
    # получение трайна и 'y' (игрик) для модели
    df = raw[~raw['cfips'].isin(blac_cfips)]
    X_train, y_train = train_and_y(df, mes_1, mes_val, train_col)
    # получение х_тест и y_тест
    X_test, y_test = x_and_y_test(raw, mes_val, train_col)
    # Создаем модель 1.858519 -0.011869
    # model = xgb.XGBRegressor(
    #     tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
    #     n_estimators=1000,
    #     learning_rate=0.0108, #важная штука
    #     # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
    #     # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
    #     max_leaves=17,
    #     #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
    #     n_jobs=2,
    # )
    model = xgb.XGBRegressor(
        tree_method="hist", # окончательный запуск без "hist". намного дольше, но точнее
        n_estimators=850,
        learning_rate=0.0071, #важная штука
        # max_depth=8, этот вариант лучше чем max_leaves=17 но в 2 раза дольше.
        # В окончательном варианте надо удалить max_leaves=17 и поставить max_depth=8
        max_leaves=17,
        #max_bin=4096, #Увеличение повышает оптимальность за счет увеличения времени вычислений.
        n_jobs=2,
    )
    model.fit(X_train, y_train)
    # Предсказываем
    y_pred = model.predict(X_test)

    # mask = raw.dcount == mes_val
    # sppol = []
    # for lag in range(1, 17):
    #     sppol.append(f'target_lag{lag}')
    # delt = 0.8
    # quan = raw.loc[mask, sppol].abs().quantile(delt, axis=1)
    # quan = quan.values
    # y_pred[y_pred > quan] = quan[y_pred > quan]
    # y_pred[y_pred < -quan] = -quan[y_pred < -quan]


    y_pred = y_pred + 1
    y_pred = raw.loc[raw.dcount == mes_val, 'mbd_lag1'] * y_pred

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
        mes_1 = 2 #
        raw, znachenie = vsia_model1(raw, mes_1, mes_val, train_col, znachenie, blac_cfips, param=0)
    #mace_blac_list(raw, start_val, mes_val)
    #print('Обучение + предсказание + обработка заняло', datetime.now() - start_time)
    # print('Значимость колонок трайна в модели')
    # znachenie.sort_values(by='importance', ascending=False, inplace=True)
    # print(znachenie)
    return raw, blac_cfips

# определяем ошибку модели для сегмента cfips с lastactive в интервале от mini до maxi
# при параметрах min_type=0, min_blac=-1, max_blac=0 определяющих блек лист
def serch_error(raw, rezult, mini, maxi, lastactive, param=0.0054):
    # создаем блек-лист cfips которых не используем в тесте
    maska = (raw['lastactive'] < mini)
    if maxi > 1:
        maska = maska|(raw['lastactive'] >= maxi)
    blac_test_cfips = raw.loc[maska, 'cfips']
    blac_test_cfips = blac_test_cfips.unique()

    # создаем блек-лист cfips который не нужен в трайне
    # if lastactive >= maxi:
    #     maska = (raw['lastactive'] < mini) | ((raw['lastactive'] > maxi) &
    #                                              (raw['lastactive'] < lastactive))
    # elif lastactive <= mini:
    #     maska = (raw['lastactive'] < lastactive)
    # else:
    #     print('Чо за ерунда?')
    #     maska = (raw['lastactive'] < mini)

    maska = (raw['lastactive'] < lastactive)

    blac_cfips = raw.loc[maska, 'cfips']
    blac_cfips = blac_cfips.unique()

    raw = new_target(raw, param)
    # здесь должен начинаться цикл по количеству лагов
    raw, train_col = build_lag(raw)  # создаем лаги c 2 и mes_1 = 4 # error 1.598877, dif_err -0.287906

    start_val = 31#33#22#6  # первый месяц валидации с которого проверяем модель
    stop_val = 31  # последний месяц валидации до которого проверяем модель
    # запускаем модель по всем месяцам
    raw, blac_cfips = modeli_po_mesiacam(raw, start_val, stop_val, train_col, blac_cfips)
    # валидация по результату обработки всех месяцев валидации в цикле
    rezult = validacia(raw, start_val, stop_val, rezult, blac_test_cfips, lastactive, mini, maxi)
    # здесь заканчиваются циклы оптимизации
    error = rezult.loc[len(rezult.index)-1,'error']
    return error, rezult

# оптимизируем сегмент cfips в интервале от mini до maxi
# составляем список из 5 лучших возможных lastactive
def serch_lastactive(raw, mini, maxi, rezult):
    potolok = 22000#mini # максимальное значение искомого параметра - lastactive
    lastactive = -1#mini
    error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
    # создаем датафрейм для хранения лучших результатов
    dflastactive = pd.DataFrame({'lastactive':[lastactive], 'error':[error]})

    lastactive = mini
    error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
    dflastactive.loc[len(dflastactive.index)] = [lastactive, error]

    lastactive = mini // 5  # минимальное начальное значение lastactive на отрезке меньше чем mini
    # первичный проход с поиском лучших результатов - минимальных ошибок
    while lastactive < potolok:
        lastactive = round(lastactive *1.2)
        # определяем ошибку модели
        # if (lastactive > maxi)| (lastactive < mini):
        error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
        dflastactive.loc[len(dflastactive.index)]=[lastactive,error]

    print('Середина serch_lastactive')
    print('mini=', mini, 'maxi=', maxi)
    rezult.sort_values(by='dif_err', inplace=True, ascending=False)
    print(rezult.head(50))
    del_shag = 2 # делитель шага; во сколько раз уменьшается шаг на каждом следующем уровне
    # множество в котором количество строк в raw соответствующее проверенным значений lastactive
    ispolzovano  = set()
    while del_shag < 100:  # цикл с уменьшением шага
        dflastactive.sort_values(by='error', inplace=True, ignore_index=True)
        dflastactive = dflastactive.loc[0:4] # оставляем 5 лучших вариантов
        for ind in range(5): # проверяем значения близкие к лучшим
            row = dflastactive.loc[ind] # строка датафрейма
            mini_shag = round(row['lastactive'] * 0.1 / del_shag)
            if mini_shag >= 1: # Проверить коэффициент
                lastactive = row['lastactive'] + mini_shag # проверяем лучшее значение + пол шага
                # if (lastactive > maxi) | (lastactive < mini):
                # количество cfips у которых lastactive больше порога; для исключения одинаковых наборов
                cfips_in = (raw['lastactive'] > lastactive).sum()
                if not (cfips_in in ispolzovano): # если такого значения lastactive еще не было
                    ispolzovano.add(cfips_in) # добавляем к использованным
                    # определяем ошибку модели
                    error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
                    dflastactive.loc[len(dflastactive.index)] = [lastactive, error]
                lastactive = row['lastactive'] - mini_shag # проверяем лучшее значение - пол шага
                # if (lastactive > maxi) | (lastactive < mini):
                cfips_in = (raw['lastactive'] > lastactive).sum()
                if not (cfips_in in ispolzovano): # если такого значения lastactive еще не было
                    ispolzovano.add(cfips_in)
                    # определяем ошибку модели
                    error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
                    dflastactive.loc[len(dflastactive.index)] = [lastactive, error]
        del_shag = del_shag * 2
    dflastactive.sort_values(by='error', inplace=True, ignore_index=True)
    lastactive = dflastactive.loc[0,'lastactive'] # получаем лучший lastactive
    print('Завершение serch_lastactive')
    print('mini=',mini,'maxi=',maxi)
    rezult.sort_values(by='dif_err', inplace=True, ascending=False)
    print(rezult.head(50))
    return lastactive, rezult # возвращаем лучший lastactive

def serch_param(raw, lastactive, mini, maxi, rezult):
    minint = 40
    maxint = 137
    shag = 16
    minerror = 1000
    ideal_param = 54
    sort2_param = 22
    spisok = list(range(minint,maxint,shag))
    while shag > 0:
        for param in spisok:
            if maxi > 1:
                error, rezult = serch_error(raw, rezult, mini, maxi, lastactive, param/10000)
            else:
                error, rezult = serch_error(raw, rezult, mini, 100000000000, lastactive, param/10000)
            if error < minerror:
                minerror = error
                sort2_param = ideal_param
                ideal_param = param
        shag = shag // 2
        spisok = []
        if ideal_param-shag > minint:
            spisok = spisok + [ideal_param-shag]
        if ideal_param+shag < maxint:
            spisok = spisok + [ideal_param+shag]
        if (sort2_param-shag != ideal_param+shag) & (sort2_param-shag > minint):
            spisok = spisok + [sort2_param-shag]
        if (sort2_param+shag != ideal_param-shag) & (sort2_param+shag < maxint):
            spisok = spisok + [sort2_param+shag]
    print('serch_param')
    print('mini=', mini, 'maxi=', maxi)
    rezult.sort_values(by='dif_err', inplace=True, ascending=False)
    print(rezult.head(50))
    ideal_param = ideal_param/10000
    return ideal_param, rezult

# оптимизируем один сегмент cfips с lastactive в интервале от mini до maxi
def optimizacia_segmenta(raw, mini, maxi, rezult):
    if maxi > 1:
        lastactive, rezult = serch_lastactive(raw, mini, maxi, rezult)
        error, rezult = serch_error(raw, rezult, mini, maxi, lastactive)
    else:
        lastactive, rezult = serch_lastactive(raw, mini, 1000000000, rezult)
        error, rezult = serch_error(raw, rezult, mini, 1000000000, lastactive)
    # param, rezult = serch_param(raw, lastactive, mini, maxi, rezult)
    error = rezult.loc[len(rezult.index) - 1, 'error']
    dif_err = rezult.loc[len(rezult.index) - 1, 'dif_err']
    dif_no_blac = rezult.loc[len(rezult.index) - 1, 'dif_no_blac']
    return lastactive, error, dif_err, dif_no_blac

# распечатываем датафрейм sedmenti
def print_sedmenti(sedmenti):
    for versia in sedmenti['versia'].unique():
        maska = sedmenti['versia'] == versia
        df = sedmenti[maska]
        print(f"Версия = {versia}. Оптимизированно на {df['optim'].mean()*100}%")
        ermean = df['error'].mean()
        dif_errmean = df['dif_err'].mean()
        print(f"error:{ermean}; dif_err:{dif_errmean}")
        print(df.head(40))

# оптимизируем все сегменты определенной версии в датафрейме sedmenti
def optimizacia_segmentacii(raw, rezult):
    versia = 1
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    sedmenti = pd.read_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno6.csv")
    df = sedmenti[(sedmenti['versia'] == versia)&(sedmenti['optim'] == 0)]
    for ind, row in df.iterrows(): # цикл по датафрейму сегменты
        # оптимизация отдельного сегмента
        rez_optim = optimizacia_segmenta(raw, row['min'], row['max'], rezult)
        lastactive, error, dif_err, dif_no_blac = rez_optim # результат оптимизации
        # внесение результатаов оптимизации в датафрейм сегменты
        sedmenti.loc[ind,'lastactive'] = lastactive
        sedmenti.loc[ind, 'error'] = error
        sedmenti.loc[ind, 'dif_err'] = dif_err
        sedmenti.loc[ind, 'dif_no_blac'] = dif_no_blac
        sedmenti.loc[ind, 'optim'] = 1
        sedmenti.to_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno6.csv", index=False)
        rezult = pd.DataFrame(columns=['lastactive', 'param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    print_sedmenti(sedmenti)

# оптимизируем все param определенной версии в датафрейме sedmenti
def optimizacia_param(raw, rezult):
    versia = 1
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    sedmenti = pd.read_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno3.csv")
    df = sedmenti[(sedmenti['versia'] == versia)&(sedmenti['optim'] == 1)]
    for ind, row in df.iterrows(): # цикл по датафрейму сегменты
        # оптимизация отдельного сегмента
        lastactive = sedmenti.loc[ind,'lastactive']
        mini = sedmenti.loc[ind,'min']
        maxi = sedmenti.loc[ind,'max']
        if maxi > 1:
            param, rezult = serch_param(raw, lastactive, mini, maxi, rezult)
            error, rezult = serch_error(raw, rezult, mini, maxi, lastactive, param)
        else:
            param, rezult = serch_param(raw, lastactive, mini, maxi, rezult)
            error, rezult = serch_error(raw, rezult, mini, 100000000000, lastactive, param)
        error = rezult.loc[len(rezult.index) - 1, 'error']
        dif_err = rezult.loc[len(rezult.index) - 1, 'dif_err']
        dif_no_blac = rezult.loc[len(rezult.index) - 1, 'dif_no_blac']
        # внесение результатаов оптимизации в датафрейм сегменты
        sedmenti.loc[ind, 'param'] = param
        sedmenti.loc[ind, 'error'] = error
        sedmenti.loc[ind, 'dif_err'] = dif_err
        sedmenti.loc[ind, 'dif_no_blac'] = dif_no_blac
        sedmenti.loc[ind, 'optim'] = 2
        sedmenti.to_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno3.csv", index=False)
        rezult = pd.DataFrame(columns=['lastactive', 'param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
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
    # granici =[60, 90, 120, 160, 220, 300, 400, 550, 750, 1050, 1450, 2300, 4000, 8000, 22000]
    granici = [60, 160, 400, 750, 1450, 2300, 22000]
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
    sedmenti = pd.DataFrame(d)
    sedmenti = kol_in_sedmenti(raw, sedmenti, versia)
    print_sedmenti(sedmenti)
    sedmenti.to_csv("C:\\kaggle\\МикроБизнес\\targ_diskretno7.csv", index=False)

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
    rezult.sort_values(by='dif_no_blac', inplace=True, ascending=False)
    print(rezult.head(22))

if __name__ == "__main__":
    pd.options.display.width = 0
    # train, test = start()
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw2.csv", index=False)
    rezult = pd.DataFrame(columns=['lastactive','param', 'kol', 'error', 'dif_err', 'dif_no_blac'])
    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv")

    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw_no_blac.csv")
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['ypred'] = 0
    #init_segmentacii(raw)
    #optimizacia_segmentacii(raw, rezult)
    optimizacia_param(raw, rezult)

