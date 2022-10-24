import pandas as pd
import hashlib
def dataprocessing():

    data1 = pd.read_csv("data_original/Graduate school admission data.csv")
    data2 = pd.read_csv("data_original/Students' Academic Performance Dataset.csv")
    data3 = pd.read_excel("data_original/“American University Data” IPEDS dataset.xlsx", sheet_name='Data')

    #calculating hash for all three datasets
    with open("data_original/Graduate school admission data.csv", 'rb') as file:
        bytes = file.read()
        hash = hashlib.md5(bytes).hexdigest();
        print(hash)

    with open("data_original/Students' Academic Performance Dataset.csv", 'rb') as file:
        bytes = file.read()
        hash = hashlib.md5(bytes).hexdigest();
        print(hash)

    with open("data_original/“American University Data” IPEDS dataset.xlsx", 'rb') as file:
        bytes = file.read()
        hash = hashlib.md5(bytes).hexdigest();
        print(hash)

    # Columns needs to drop from data3 which are irrelevant
    columns_need_dropping_data1 = ['NationalITy', 'PlaceofBirth']
    data2 = data2.drop(columns_need_dropping_data1, axis=1)

    columns_need_dropping = ['Tuition and fees, 2010-11', 'Tuition and fees, 2011-12', 'Tuition and fees, 2012-13', 'Tuition and fees, 2013-14', 'Total price for out-of-state students living on campus 2013-14', 'State abbreviation', 'FIPS state code', 'Geographic region', 'Sector of institution', 'Level of institution', 'Control of institution', 'Historically Black College or University', 'Tribal college', 'Degree of urbanization (Urban-centric locale)', 'Carnegie Classification 2010: Basic', 'Endowment assets (year end) per FTE enrollment (GASB)', 'Endowment assets (year end) per FTE enrollment (FASB)', 'ZIP code', 'Latitude location of institution', 'Longitude location of institution']
    data3 = data3.drop(columns_need_dropping, axis=1)

    # Getting the columns names which has missing values
    check_missing_sum_data1 = pd.isnull(data1).sum().sum()
    check_missing_sum_data2 = pd.isnull(data2).sum().sum()
    check_missing_sum_data3 = pd.isnull(data3).sum().sum()

    # We will fill the missing value with the value from precedent row
    if check_missing_sum_data1 > 0:
        missing_value_data1 = data1.columns[data1.isnull().any()]
        for i in missing_value_data1:
            data1[i] = data1[i] = data1[i].fillna(method='bfill')

    if check_missing_sum_data2 > 0:
        missing_value_data2 = data2.columns[data2.isnull().any()]
        for i in missing_value_data2:
            data2[i] = data2[i].fillna(method='bfill')

    if check_missing_sum_data3 > 0:
        missing_value_data3 = data3.columns[data3.isnull().any()]
        for i in missing_value_data3:
            data3[i] = data3[i] = data3[i].fillna(method='bfill')


    data1.to_csv('data_processed/Graduate school admission data.csv')
    data2.to_csv('data_processed/Students Academic Performance Dataset.csv')
    data3.to_csv('data_processed/“American University Data” IPEDS dataset.csv')

dataprocessing()