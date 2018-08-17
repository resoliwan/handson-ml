from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
    random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(
      splitter="random",
      random_state=42
      ),
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
    random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=32)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
  print(name, score)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(
      splitter="random",
      random_state=42),
    n_estimators=500,
    max_samples=100,
    n_jobs=-1,
    random_state=42,
    oob_score=True)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)


from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(),
    n_estimators=200,
    learning_rate=0.6
    )

ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
accuracy_score(y_test, y_pred)

import numpy as np

np.random.seed(42)

X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2)
tree_reg1.fit(X, y)
y2 = y - tree_reg1.predict(X)

tree_reg2 = DecisionTreeRegressor(max_depth=2)
tree_reg2.fit(X, y2)
y3 = y2 - tree_reg2.predict(X)

tree_reg3 = DecisionTreeRegressor(max_depth=2)
tree_reg3.fit(X, y3)

X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

from sklearn.ensemble import GradientBoostingRegressor

grbt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1)
grbt.fit(X, y)

grbt.predict(X_new)


grbt = GradientBoostingRegressor(max_depth=2, n_estimators=120)



