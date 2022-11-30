from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from IPython.display import HTML

data = pd.DataFrame({
    'X_1': [1, 1, 1, 0, 0, 0, 0, 1],
    'X_2': [0, 0, 0, 1, 0, 0, 0, 1],
    'Y': [1, 1, 1, 1, 0, 0, 0, 0]
})
#print(data)
clf = tree.DecisionTreeClassifier(criterion='entropy')
X = data[['X_1', 'X_2']]
y = data.Y
clf.fit(X, y)
graph = Source(tree.export_graphviz(clf, out_file=None,
                                    feature_names=list(X),
                                    class_names=['Negative', 'Positive'],
                                    filled=True))
display(SVG(graph.pipe(format='svg')))
