import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold

from servise_ds import okno
iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y)

train = pd.DataFrame(X_train)
train['y'] = y_train
test = pd.DataFrame(X_test)
y = train.pop('y')
X = train

param_grid = {
    "max_depth": range(1,10),
    "min_samples_split": range(2,10),
    "min_samples_leaf": range(1,10)
}
tree = DecisionTreeClassifier()
search = GridSearchCV(estimator=tree, cv=5, param_grid=param_grid)
#search = RandomizedSearchCV(estimator=tree, cv=5, param_distributions=param_grid)
search.fit(X, y)
best_tree = search.best_estimator_
predictions = search.predict(test)