import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utilities import bootstraps

def test_clf(clf, test_set, test_label, *argv):
    y_val = test_label
    test_data = test_set
    y_pred = clf.predict(test_data)
    #y_prob = clf.best_estimator_.predict_proba(test_data)
    y_prob = clf.predict_proba(test_data)
    
    if argv != False:
        print(classification_report(y_val, y_pred))
        print('the accuracy score of the classifier is {}'.format(accuracy_score(y_val, y_pred)))
        bootstraps(y_val, y_pred,function='ACC')
        print('the kappa score of the classifier is {}'.format(cohen_kappa_score(y_val, y_pred, weights = 'linear')))
        bootstraps(y_val, y_pred,function='kappa')

    if len(np.unique(y_val)) == 2:
        scores = y_prob[:,1]
        fpr, tpr, threshold = roc_curve(y_val, scores)
        auc = roc_auc_score(y_val, scores)
        print('AUC: {}'.format(auc))
        bootstraps(y_val,scores,function = 'AUC')
        #RocCurveDisplay.from_estimator(clf, test_data, y_val, )
    else:
        fpr, tpr, threshold = [], [], []
    
    #ConfusionMatrixDisplay.from_estimator(clf, test_data, y_val, xticks_rotation="vertical")
    cm = confusion_matrix(test_label, y_pred)
    #cm_display = ConfusionMatrixDisplay(cm).plot()
