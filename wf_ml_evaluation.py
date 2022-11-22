import pandas as pd
import numpy as np
import wf_ml_training
import wf_ml_prediction
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def model_RandomForestClassifier(train_X, train_Y, test_X, test_Y, n_estimators=1000):
    classifier = RandomForestClassifier(n_estimators=n_estimators)
    #The model building takes time, to save time you can comment below line and uncomment classifier = pickle.load(open(classifier, 'rb')) line to read model and predict on it
    classifier.fit(train_X, train_Y)
    #filename = 'models/RandomForestClassifier.sav'
    # pickle.dump(classifier, open(filename, 'wb'))
    #classifier = pickle.load(open(filename, 'rb'))
    y_pred = classifier.predict(test_X)
    accuracy = round(accuracy_score(test_Y, y_pred) * 100, 2)
    print("RandomForestClassifier accuracy : ", accuracy)
    print()
    with open('evaluation/summary.txt', 'a') as f:
        f.write("\n\nRandomForestClassifier algorithm : \n")
        f.write("\nAccuracy : ")
        f.write(str(accuracy))
        f.write("\nRoot mean squared error :")
        f.write(str(mean_squared_error(y_pred, test_Y) * 100))
        f.write("\nConfusion metrix : \n")
        f.write(str(confusion_matrix(y_pred, test_Y)))
        f.write("\nF1 score : ")
        f.write(str(f1_score(test_Y, y_pred, average='micro')))

    array = confusion_matrix(y_pred, test_Y)
    df_cm = pd.DataFrame(array, index=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                         columns=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax = plt.axes()
    ax.set_title('RandomForest confusion metrix heatmap')
    swarm_plot = sn.heatmap(df_cm, annot=True, ax=ax)
    fig = swarm_plot.get_figure()
    fig.savefig("visuals/RandomForest_confusion_metrix")

def model_SVM(train_X, train_Y, test_X, test_Y):
    classifier = SVC(kernel='linear', random_state=0)
    # The model building takes time, to save time you can comment below line and uncomment classifier = pickle.load(open(classifier, 'rb')) line to read model and predict on it
    classifier.fit(train_X, train_Y)
    #filename = 'models/SVC.sav'
    # pickle.dump(classifier, open(filename, 'wb'))
    #classifier = pickle.load(open(filename, 'rb'))
    y_pred = classifier.predict(test_X)
    accuracy = round(accuracy_score(test_Y, y_pred) * 100, 2)
    print("SVM accuracy : ", accuracy)
    print()
    with open('evaluation/summary.txt', 'a') as f:
        f.write("\n\nSupport vector machine(SVM) algorithm : \n")
        f.write("\nAccuracy : ")
        f.write(str(accuracy))
        f.write("\nRoot mean squared error :")
        f.write(str(mean_squared_error(y_pred, test_Y) * 100))
        f.write("\nConfusion metrix : \n")
        f.write(str(confusion_matrix(y_pred, test_Y)))
        f.write("\nF1 score : ")
        f.write(str(f1_score(test_Y, y_pred, average='micro')))

    array = confusion_matrix(y_pred, test_Y)
    df_cm = pd.DataFrame(array, index=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                         columns=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax = plt.axes()
    ax.set_title('Support vector machine(SVM) confusion metrix heatmap')
    swarm_plot = sn.heatmap(df_cm, annot=True, ax=ax)
    fig = swarm_plot.get_figure()
    fig.savefig("visuals/SVC_confusion_metrix")


def model_KNeighborsClassifier(train_X, train_Y, test_X, test_Y, k):
    classifier = KNeighborsClassifier(n_neighbors=k)
    # The model building takes time, to save time you can comment below line and uncomment classifier = pickle.load(open(classifier, 'rb')) line to read model and predict on it
    classifier.fit(train_X, train_Y)
    #filename = 'models/KNeighborsClassifier.sav'
    # pickle.dump(classifier, open(filename, 'wb'))
    #classifier = pickle.load(open(filename, 'rb'))
    y_pred = classifier.predict(test_X)
    accuracy = round(accuracy_score(test_Y, y_pred) * 100, 2)
    print("KNeighborsClassifier accuracy : ", accuracy)
    print()
    with open('evaluation/summary.txt', 'a') as f:
        f.write("\n\nKNeighborsClassifier algorithm : \n")
        f.write("\nAccuracy : ")
        f.write(str(accuracy))
        f.write("\nRoot mean squared error :")
        f.write(str(mean_squared_error(y_pred, test_Y) * 100))
        f.write("\nConfusion metrix : \n")
        f.write(str(confusion_matrix(y_pred, test_Y)))
        f.write("\nF1 score : ")
        f.write(str(f1_score(test_Y, y_pred, average='micro')))


def ml_evaluation():
    df_1 = pd.read_pickle("data_processed/processed.pkl")
    images = df_1['Images']
    labels = df_1['labels']
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    label_wm = le.transform(labels)
    train_X, test_X, train_Y, test_Y = model_selection.train_test_split(images, label_wm, test_size=0.2)
    wf_ml_training.ml_training(train_X, train_Y)
    wf_ml_prediction.ml_prediction(test_X, test_Y)

    model_RandomForestClassifier(train_X, train_Y, test_X, test_Y, n_estimators=1000)
    model_KNeighborsClassifier(train_X, train_Y, test_X, test_Y, k=4)
    model_SVM(train_X, train_Y, test_X, test_Y)


ml_evaluation()