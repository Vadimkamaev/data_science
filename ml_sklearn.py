import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# МОДУЛИ БИБЛИОТЕКИ sklearn:

# sklearn.model_selection - модели разделения данных на часть для обучения и часть для теста
# train_test_split одноразовое разделение датафрейма на часть для обучения и для проверки
# cross_val_score - кросс-валидатор - заданное количество разделений и проверок;
# GridSearchCV - перекрестная проверка (используется в модуле оптимизации)
# K-Folds - кросс-валидатор

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold

# sklearn.ensemble - ансабли;
# RandomForestRegressor - Случайный лес; GradientBoostingClassifier - градиентный бустинг
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

# sklearn.metrics методы оценки прогнозов; mean_squared_error - измерение расстояния между моделью и истиной
# accuracy_score - доля выборок, правильно спрогнозированных; roc_auc_score - площадь под кривой
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

# sklearn.linear_model - линейные модели; LogisticRegression - логистическая регрессия
from sklearn.linear_model import LogisticRegression

# решающее дерево Decision Tree
from sklearn import tree


#из Титаника
from sklearn.preprocessing import LabelEncoder # перекодировка

# модели для бинарной класификации
class Cbin:

    # логистическая регрессия, обучение линейной модели
    def line(self, data, y): # датафрейм исходных данных, y - то что надо получить
        lr = LogisticRegression() # создание модели
        lr.fit(data, y) # обучение модели
        lr_preds = lr.predict(data) # предсказание исходя из модели
        lr_preds_proba = lr.predict_proba(data)[:, 1] # оценка вероятности

        print('accuracy', accuracy_score(y, lr_preds)) # оценка похожести предсказания и 'y'
        print('roc_auc', roc_auc_score(y, lr_preds_proba)) # другая оценка
        # кросвалидация; параметры: модель, датафрейм, результирующие данные, кол. разбиений, метод оценки
        cross_val_scores = cross_val_score(lr, data, y, cv=5, scoring='roc_auc')
        print(data.head(1))
        print(cross_val_scores)
        print(np.mean(cross_val_scores))
        print(lr.coef_) #0.8754966485996073
        return lr

    # решающее дерево Decision Tree
    def tree(self, data, y, pechat=2): # датафрейм исходных данных, y - то что надо получить
        model = tree.DecisionTreeClassifier(criterion='entropy') # создание модели
        model.fit(data, y) # обучение модели
        model_preds = model.predict(data) # предсказание исходя из модели
        model_preds_proba = model.predict_proba(data)[:, 1] # оценка вероятности
        if pechat==2:
            print('accuracy', accuracy_score(y, model_preds)) # оценка похожести предсказания и 'y'
            print('roc_auc', roc_auc_score(y, model_preds_proba)) # другая оценка
        cross_val_scores = cross_val_score(model, data, y, cv=5, scoring='accuracy')
        if pechat == 2:
            print(cross_val_scores)
        tochnost = np.mean(cross_val_scores)
        if pechat > 0:
            print(tochnost)
        return model, tochnost

    # градиентный бустинг Gradient Boosting regression
    def gbdt_mod(self, data, y, pechat=2): # датафрейм исходных данных, y - то что надо получить
        gbdt = GradientBoostingClassifier() # создание модели
        gbdt.fit(data, y) # обучение модели
        gbdt_preds = gbdt.predict(data) # предсказание исходя из модели
        gbdt_preds_proba = gbdt.predict_proba(data)[:, 1] # оценка вероятности
        if pechat==2:
            print('accuracy', accuracy_score(y, gbdt_preds)) # оценка похожести предсказания и 'y'
            print('roc_auc', roc_auc_score(y, gbdt_preds_proba)) # другая оценка
            # кросвалидация; параметры: модель, датафрейм, результирующие данные, кол. разбиений, метод оценки
        cross_val_scores = cross_val_score(gbdt, data, y, cv=5, scoring='accuracy')
        if pechat == 2:
            print(cross_val_scores)
        m = np.mean(cross_val_scores)
        if pechat > 0:
            print(m)
        return gbdt, m

    # оптимизация параметров градиентного бустинга
    def gbdt_optimizm(self, data, y): # датафрейм исходных данных, y - то что надо получить
        param_grid = {
            "max_depth": [2, 3],
            "n_estimators": [50, 100, 150],
            # "learning_rate": [0.01, 0.05, 0.1],
            # "min_child_weight":[4,5,6],
            # "subsample": [0.8, 0.9, 1]
        }
        gbdt = GradientBoostingClassifier()
        kfold = KFold(n_splits=5, shuffle=True, random_state=123)
        CV_gbdt = GridSearchCV(estimator=gbdt, param_grid=param_grid,
                               scoring='roc_auc', cv=kfold, verbose=1000)
        CV_gbdt.fit(data, y)
        print(CV_gbdt.best_params_)
        print(CV_gbdt.best_score_)
        return CV_gbdt

bin = Cbin()

# определение значения вещественной переменной от значений других переменных
class Creg:

    # случайный лес
    def les(self, df):
        # Деление данных на train / test
        target = df.target
        data = df.drop('target', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=322, test_size=0.1)
        # Создаем модель
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=322)
        # Обучаем
        rf.fit(X_train, y_train)
        # Качество модели на той выборке, на которой она обучалась
        print('Train Accuracy:', mean_squared_error(y_train, rf.predict(X_train)))
        # Качество на тестовой выборке
        print('Validation Accuracy:', mean_squared_error(y_test, rf.predict(X_test)))
        self.les_priznaki()
        return rf

    # Отбор признаков которые сильнее влияют на результат
    def les_priznaki(self, data, rf):
        # Упорядычиваем наши фичи по значениям весов, от самой полезной к самой бесполезной
        df_importances = sorted(list(zip(data.columns, rf.feature_importances_.ravel())), key=lambda tpl: tpl[1],
                                reverse=True)
        # Создаем табличку, в которой будет показан признак и его вес
        df_importances = pd.DataFrame(df_importances, columns=['feature', 'importance'])
        # Нумируем колонки, чтобы не путать их
        df_importances = df_importances.set_index('feature')
        # Создаем график, чтобы было нагляднее
        df_importances.plot(kind='bar', figsize=(15, 3))
        # Рисуем график
        plt.show()
        # Выводим табличку
        print(df_importances.head(10))





