import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

if __name__ == '__main__':
    loans = pd.read_csv('../data/loan_data.csv')
    print(loans.info())
    cat_feats = ['purpose']
    final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
    print(final_data.info())

    X = final_data.drop('not.fully.paid',axis=1)
    y = final_data['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred_tree = dtree.predict(X_test)
    print('Decision Tree Results: \n', classification_report(y_test, y_pred_tree))
    print(confusion_matrix(y_test, y_pred_tree), '\n')

    rforest = RandomForestClassifier(n_estimators=600)
    rforest.fit(X_train, y_train)
    y_pred_forest = rforest.predict(X_test)
    print('Random Forest Results: \n', classification_report(y_test, y_pred_forest))
    print(confusion_matrix(y_test, y_pred_forest))



