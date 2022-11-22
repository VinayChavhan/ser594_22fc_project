import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

def predict_XGBoost(test_X, test_Y):
    loaded_model = pickle.load(open("models/XGBoost.sav", 'rb'))
    prediction = loaded_model.predict(test_X)
    accuracy = round(accuracy_score(test_Y, prediction)*100,2)
    print("XGBoost accuracy : ", accuracy)
    print()
    with open('evaluation/summary.txt', 'w') as f:
        f.write("XGBoost algorithm : \n")
        f.write("\nAccuracy : ")
        f.write(str(accuracy))
        f.write("\nRoot mean squared error :")
        f.write(str(mean_squared_error(prediction, test_Y) * 100))
        f.write("\nConfusion metrix : \n")
        f.write(str(confusion_matrix(prediction, test_Y)))
        f.write("\nF1 score : ")
        f.write(str(f1_score(test_Y, prediction, average='micro')))

    array = confusion_matrix(prediction, test_Y)
    df_cm = pd.DataFrame(array, index=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                         columns=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax = plt.axes()
    ax.set_title('XGBoost confusion metrix heatmap')
    swarm_plot = sn.heatmap(df_cm, annot=True, ax=ax)
    fig = swarm_plot.get_figure()
    fig.savefig("visuals/XGBoost_confusion_metrix")


def ml_prediction(test_X, test_Y):
    predict_XGBoost(test_X, test_Y)