import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    df = pd.read_csv('../data/kyphosis.csv')
    toggle = False
    if toggle:
        print(df.head())
        plot1 = sns.pairplot(df, hue='Kyphosis', palette='Set1')
        plt.savefig('../imgs/pairplot_kyphosis.png')
        plt.show();
    X = df.drop('Kyphosis', axis=1)
    y = df['Kyphosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=101)
    # First see a single tree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred_tree = dtree.predict(X_test)
    # Check out a forest
    rforest = RandomForestClassifier(n_estimators=200)
    rforest.fit(X_train, y_train)
    y_pred_forest = rforest.predict(X_test)
    # View the results
    print('Decision Tree Results: \n', classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree), '\n')

    print('Random Forest Results: \n', classification_report(y_test, y_pred_forest))
    print(confusion_matrix(y_test, y_pred_forest))