import pandas as pd
import numpy as np
import torch
import pickle
from PIL import Image
from sklearn import preprocessing
import wf_visualization

def dataprocessing():
    df = pd.read_pickle("data_original/LSWMD.pkl")
    print("Head", df.head())
    print("Tail", df.tail())
    DEVICE = torch.device("cpu")

    def find_dim(x):
        dim0 = np.size(x, axis=0)
        dim1 = np.size(x, axis=1)
        return dim0, dim1
    df = df.drop(['waferIndex'], axis=1)
    df['waferMapDim'] = df.waferMap.apply(find_dim)

    # extract specific dim
    def subwafer(data, Dim0, Dim1):
        sw = torch.ones((1, Dim0, Dim1)).int()
        label = list()
        Dim0 = np.size(sw, axis=1)
        Dim1 = np.size(sw, axis=2)
        sub_df = data.loc[data['waferMapDim'] == (Dim0, Dim1)]
        sw = sw.to(DEVICE)
        for i in range(len(sub_df)):
            waferMap = torch.from_numpy(sub_df.iloc[i, :]['waferMap'].reshape(1, Dim0, Dim1)).int()
            waferMap = waferMap.to(DEVICE)
            sw = torch.cat([sw, waferMap])
            label.append(sub_df.iloc[i, :]['failureType'][0][0])
        x = sw[1:]
        y = np.array(label).reshape((-1, 1))
        del waferMap, sw
        return x, y

    def rgb_sw(x):
        Dim0 = np.size(x, axis=1)
        Dim1 = np.size(x, axis=2)
        new_x = np.zeros((len(x), Dim0, Dim1, 3))
        x = torch.unsqueeze(x, -1)
        x = x.to(torch.device('cpu'))
        x = x.numpy()
        for w in range(len(x)):
            for i in range(Dim0):
                for j in range(Dim1):
                    new_x[w, i, j, int(x[w, i, j])] = 1
        return new_x

    def resize(x):
        rwm = torch.ones((1, 26, 26, 3))
        for i in range(len(x)):
            rwm = rwm.to(torch.device('cpu'))
            a = Image.fromarray(x[i].astype('uint8')).resize((26, 26))
            a = np.array(a).reshape((1, 26, 26, 3))
            a = torch.from_numpy(a).float()
            a = a.to(torch.device('cpu'))
            rwm = torch.cat([rwm, a])
        x = rwm[1:]
        del rwm
        return x

    def _to_one_hot(y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    df['failureNum'] = df.failureType
    df['trainTestNum'] = df.trianTestLabel
    mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                    'Near-full': 7, 'none': 8}
    mapping_traintest = {'Training': 0, 'Test': 1}
    df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

    tol_wafers = df.shape[0]

    df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
    df_withlabel = df_withlabel.reset_index()
    defined_df = df.loc[df['failureType'] != 0]  # failureType !=0
    defined_withpattern = defined_df[(defined_df['failureType'] != 'none')]
    defined__withpattern = defined_withpattern.reset_index()  # patterned index.
    defined_nonpattern = defined_df[(defined_df['failureType'] == 'none')]  # nonpatterned index
    defined_df.shape[0], defined__withpattern.shape[0], defined_nonpattern.shape[0]

    wf_visualization.visualize_defect_frequency(tol_wafers, df_withlabel, defined_withpattern,defined_nonpattern)


    x, y = subwafer(defined_df, 26, 26)
    x1, y1 = subwafer(defined_df, 25, 27)
    x2, y2 = subwafer(defined_df, 30, 34)
    print(x.shape, y.shape, x1.shape, y1.shape, x2.shape, y2.shape)

    faulty_case = np.unique(y)
    print('Faulty case list : {}'.format(faulty_case))

    rgb_x0 = rgb_sw(x1)  # about 8s each line.
    rgb_x1 = rgb_sw(x2)

    resized_x0 = resize(rgb_x0)
    resized_x1 = resize(rgb_x1)

    resized_wm = torch.cat([resized_x0, resized_x1])
    label_wm = np.concatenate((y1, y2))  # concatenate To use all data.

    resized_wm_2d = np.reshape(resized_wm, (len(resized_wm), -1))

    final_store = {"Images": resized_wm_2d, 'labels': label_wm}

    with open("data_processed/processed.pkl", "wb") as file:
        pickle.dump(final_store, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()


dataprocessing()