import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec

def visualization():
    df_1 = pd.read_pickle("data_processed/processed.pkl")
    images = df_1['Images']
    labels = df_1['labels']
    # le = preprocessing.LabelEncoder()
    # le.fit(label_wm)
    # lables_name = le.inverse_transform(labels)
    images_4d = np.reshape(images, (len(images), 26, 26, 3))

    x = images_4d
    y = labels
    fig, ax = plt.subplots(3, 3, figsize=(5, 6))
    ax[0][0].imshow(x[np.where(y == 'Center')[0][0]].to('cpu'))
    ax[0][0].set_title("Center")
    ax[0][0].axis("off")
    ax[0][1].imshow(x[np.where(y == 'Donut')[0][2]].to('cpu'))
    ax[0][1].axis("off")
    ax[0][1].set_title("Donut")
    ax[0][2].imshow(x[np.where(y == 'Edge-Loc')[0][100]].to('cpu'))
    ax[0][2].axis("off")
    ax[0][2].set_title("Edge-Loc")
    ax[1][0].imshow(x[np.where(y == 'Edge-Ring')[0][20]].to('cpu'))
    ax[1][0].axis("off")
    ax[1][0].set_title("Edge-Ring")
    ax[1][1].imshow(x[np.where(y == 'Loc')[0][1]].to('cpu'))
    ax[1][1].axis("off")
    ax[1][1].set_title("Loc")
    ax[1][2].imshow(x[np.where(y == 'Random')[0][20]].to('cpu'))
    ax[1][2].axis("off")
    ax[1][2].set_title("Random")
    ax[2][0].imshow(x[np.where(y == 'Scratch')[0][20]].to('cpu'))
    ax[2][0].axis("off")
    ax[2][0].set_title("Scratch")
    ax[2][1].imshow(x[np.where(y == 'Near-full')[0][1]].to('cpu'))
    ax[2][1].axis("off")
    ax[2][1].set_title("Near-full")
    ax[2][2].imshow(x[np.where(y == 'none')[0][0]].to('cpu'))
    ax[2][2].axis("off")
    ax[2][2].set_title("None")
    fig.savefig("visuals/type_of_defects.png")

def visualize_defect_frequency(tol_wafers, df_withlabel, df_withpattern, df_nonpattern):
    fig = plt.figure(figsize=(20, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    no_wafers = [tol_wafers - df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

    colors = ['blue', 'green', 'red']
    explode = (0.1, 0, 0)  # explode 1st slice
    labels = ['no-label', 'label and pattern', 'label and non-pattern']
    ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    uni_pattern = np.unique(df_withpattern.failureNum, return_counts=True)
    labels2 = ['', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
    ax2.bar(uni_pattern[0], uni_pattern[1] / df_withpattern.shape[0], color='green', align='center', alpha=0.9)
    ax2.set_title("failure type frequency")
    ax2.set_ylabel("% of pattern wafers")
    ax2.set_xticklabels(labels2)

    fig.savefig("visuals/defect_frequncey.png")

visualization()