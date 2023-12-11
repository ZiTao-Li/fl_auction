from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

def base_train(train_x, train_y):
    clf = LogisticRegression(random_state=0, max_iter=2000, C=0.0001, verbose=0)\
        .fit(train_x, train_y)
    print("training done.")
    return clf


def base_test_auc(clf, test_x, test_y):
    y_pred = clf.predict_proba(test_x)[:, 1]
    auc = roc_auc_score(test_y, y_pred)
    return auc

def base_test_f1(clf, test_x, test_y):
    y_pred = clf.predict(test_x)
    f1 = f1_score(test_y, y_pred)
    return f1