import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import wf_ml_training
import wf_ml_prediction
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, KFold


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu', input_shape=(26, 26, 3)))
    model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.3))
    model.add(Conv2D(256, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(256, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def model_CNN(train_X, train_Y, test_X, test_Y):
    print("*****CNN********")
    # train_X = train_X.numpy()
    # train_Y = train_Y.numpy()
    # test_X = test_X.numpy()
    # test_Y = test_Y.numpy()
    # train_Y = train_Y.reshape((train_Y.shape[0], train_Y.shape[2]))
    # test_Y = test_Y.reshape((test_Y.shape[0], test_Y.shape[2]))
    model = Sequential()
    model.add(Conv2D(32, (4, 4), activation='relu', input_shape=(26, 26, 3)))
    model.add(Conv2D(32, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.3))
    model.add(Conv2D(256, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(256, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(Conv2D(128, (4, 4), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(.3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(9, activation='sigmoid'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    history = model.fit(train_X, train_Y,
                        batch_size=500, epochs=100,
                        validation_data=(test_X, test_Y), callbacks=callbacks)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    fig.savefig("visuals/CNN_evalution.png")

    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=1024, verbose=2)
    # 3-Fold Crossvalidation
    kfold = KFold(n_splits=4, shuffle=True, random_state=2019)
    results = cross_val_score(model, train_X, train_Y, cv=kfold)
    # Check 3-fold model's mean accuracy
    print('Simple CNN Cross validation score : {:.4f}'.format(np.mean(results)))

def model_GredientBoost(trainnew1, train_Y1, testnew1, test_Y1):
    print("*****ADA Boost********")
    clf_ada = GradientBoostingClassifier(n_estimators=5)
    clf_ada.fit(trainnew1, train_Y1)
    y_pred = clf_ada.predict(testnew1)
    print("Train data accuracy:", accuracy_score(y_true=train_Y1, y_pred=clf_ada.predict(trainnew1)))
    print("Test data accuracy:", accuracy_score(y_true=test_Y1, y_pred=y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test_Y1, y_pred))

    print("Classification Report")
    print(classification_report(test_Y1, y_pred))

def model_BaggingEnsembleModel(trainnew1, train_Y1, testnew1, test_Y1):
    print("*****BAGGING ENSEMBLE Boost********")
    clf_ada = BaggingClassifier(n_estimators=5)
    clf_ada.fit(trainnew1, train_Y1)
    y_pred = clf_ada.predict(testnew1)
    print("Train data accuracy:", accuracy_score(y_true=train_Y1, y_pred=clf_ada.predict(trainnew1)))
    print("Test data accuracy:", accuracy_score(y_true=test_Y1, y_pred=y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test_Y1, y_pred))

    print("Classification Report")
    print(classification_report(test_Y1, y_pred))

def model_AdaBoost(trainnew1, train_Y1, testnew1, test_Y1):
    print("*****ADA Boost********")
    clf_ada = AdaBoostClassifier(n_estimators=3)
    clf_ada.fit(trainnew1, train_Y1)
    y_pred = clf_ada.predict(testnew1)
    print("Train data accuracy:", accuracy_score(y_true=train_Y1, y_pred=clf_ada.predict(trainnew1)))
    print("Test data accuracy:", accuracy_score(y_true=test_Y1, y_pred=y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test_Y1, y_pred))

    print("Classification Report")
    print(classification_report(test_Y1, y_pred))

def model_RandomForestClassifier(train_X, train_Y, test_X, test_Y, n_estimators=1000):
    print("*****Random Forest********")
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

def model_DecisionTree(train_X, train_Y, test_X, test_Y):
    print("*****Decision Tree********")
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_X, train_Y)

    # Predict Accuracy Score
    y_pred = clf.predict(test_X)
    print("Train data accuracy:", accuracy_score(y_true=train_Y, y_pred=clf.predict(train_X)))
    print("Test data accuracy:", accuracy_score(y_true=test_Y, y_pred=y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(test_Y, y_pred))

    print("Classification Report")
    print(classification_report(test_Y, y_pred))

def model_SVM(train_X, train_Y, test_X, test_Y):
    print("*****SVM********")
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
    print("*****KNN********")
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

    # train_X1 = train_X.numpy()
    # test_X1 = test_X.numpy()
    trainnew1 = train_X.reshape(12131, 2028)
    testnew1 = test_X.reshape(3033, 2028)

    model_DecisionTree(trainnew1, train_Y, testnew1, test_Y)
    model_AdaBoost(trainnew1, train_Y, testnew1, test_Y)
    model_GredientBoost(trainnew1, train_Y, testnew1, test_Y)
    model_BaggingEnsembleModel(trainnew1, train_Y, testnew1, test_Y)

    # model_CNN(train_X, train_Y, test_X, test_Y)



ml_evaluation()