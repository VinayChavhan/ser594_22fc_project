import pandas as pd
import matplotlib.pyplot as plt
def visualization():
    data1 = pd.read_csv("data_processed/Graduate school admission data.csv")
    data2 = pd.read_csv("data_processed/Students Academic Performance Dataset.csv")
    data3 = pd.read_csv('data_processed/“American University Data” IPEDS dataset.csv')

    print(data1.keys())

    data1_quantitave_columns = ['gre', 'gpa', 'rank']
    data2_quanlitative_columns = ['StageID', 'ParentschoolSatisfaction']
    data3_quantitave_columns = ['Applicants total', 'Admissions total']

    with open('data_processed/summary.txt', 'w') as f:
        f.write("Quantitative Columns : \n")
        for i in data1_quantitave_columns:
            f.write("\n" + i + " : ")
            f.write("\nmin  : " + str(data1[i].min()))
            f.write("\nmax : " + str(data1[i].max()))
            f.write("\nmedian : " + str(data1[i].median()))
            f.write("\n")

        f.write("\n")
        for i in data3_quantitave_columns:
            f.write("\n" + i + " : ")
            f.write("\nmin  : " + str(data3[i].min()))
            f.write("\nmax : " + str(data3[i].max()))
            f.write("\nmedian : " + str(data3[i].median()))
            f.write("\n")

        f.write("\n")
        f.write("Qualitative Columns : \n")

        for i in data2_quanlitative_columns:
            f.write("\n" + i +" : ")
            f.write("\nNumber of Categories :"+ str(data2[i].unique()))
            f.write("\nMost Frequent category :"+ str(data2[i].max()))
            f.write("\nLeast Frequent category :" + str(data2[i].min()))
            f.write("\n")
        f.close()

    with open('data_processed/correlations.txt', 'w') as f:
        f.write("Correlation Matrix for Graduate school admission data : \n\n")
        correlation_data1 = data1[data1_quantitave_columns].corr()
        correlation_data3 = data3[data3_quantitave_columns].corr()
        correlation_data1 = round(correlation_data1, 3)
        correlation_data3 = round(correlation_data3, 3)
        f.write(correlation_data1.to_string(header=True, index=True))
        f.write("\n\nCorrelation Matrix for American University Data IPEDS dataset : \n\n")
        f.write(correlation_data3.to_string(header=True, index=True))
        f.close()

    fig, ax = plt.subplots()
    ax.set(title='GRE and GPA Plot : Quantitative data plot 1',
           ylabel='GPA', xlabel='GRE')
    plt.scatter(data1['gre'], data1['gpa'])
    ax.legend()
    fig.savefig("visuals/GRE_and_GPA_scatter_plot_quantative_plot1.png")

    fig, ax = plt.subplots()
    ax.set(title='GRE and Rank Plot : Quantitative data plot 2',
           ylabel='Rank', xlabel='GRE')

    plt.scatter(data1['gre'], data1['rank'])
    ax.legend()
    fig.savefig("visuals/GRE_and_Rank_scatter_plot_quantative_plot2.png")

    fig, ax = plt.subplots()
    ax.set(title='GPA and Rank Plot - Quantitative data plot 3',
           ylabel='Rank', xlabel='GPA')

    plt.scatter(data1['gpa'], data1['rank'])
    ax.legend()
    fig.savefig("visuals/GPA_and_Rank_scatter_plot_quantative_plot3.png")

    fig, ax = plt.subplots()
    #fig.suptitle('StageIDs', fontsize=20, color="black")
    ax.set(xlabel='StageID', ylabel='Frequency')
    plt.hist(data2["StageID"], bins=10)
    ax.legend()
    fig.savefig("visuals/StageID_hist_qualitative_plot1.png")

    fig, ax = plt.subplots()
    #fig.suptitle('StageIDs', fontsize=20, color="black")
    ax.set(xlabel='ParentschoolSatisfaction', ylabel='Frequency')
    plt.hist(data2["ParentschoolSatisfaction"], bins=10)
    ax.legend()
    fig.savefig("visuals/ParentschoolSatisfaction_hist_qualitative_plot1.png")
    print("vinay")
    visualization_df = None


visualization()