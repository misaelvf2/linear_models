import pandas as pd
import numpy as np

def process_breast_cancer_data():
    df = pd.read_csv("data/breast-cancer-wisconsin.data", header=None, na_values=['?'])
    df = df.fillna(np.random.randint(1, 11))
    df.columns = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
                  'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    df['class'] = df['class'].apply(lambda x: 1 if x == 4 else -1) # WATCH OUT FOR THIS!!!!
    normalized_df = (df.iloc[:, 1:-1] - df.iloc[:, 1:-1].mean()) / df.iloc[:, 1:-1].std()
    normalized_df.insert(0, 'id', df['id'])
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df

def process_glass_data():
    df = pd.read_csv("data/glass.data", header=None)
    df.columns = ['id', 'ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'class']
    normalized_df = (df.iloc[:, 1:-1] - df.iloc[:, 1:-1].mean()) / df.iloc[:, 1:-1].std()
    normalized_df.insert(0, 'id', df['id'])
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df

def process_iris_data():
    df = pd.read_csv("data/iris.data", header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df['class'] = df['class'].apply(convert_iris_to_numerical)
    normalized_df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df

def convert_iris_to_numerical(x):
    if x == 'Iris-setosa':
        return 1
    elif x == 'Iris-versicolor':
        return 2
    elif x == 'Iris-virginica':
        return 3

def process_soybean_data():
    df = pd.read_csv("data/soybean-small.data", header=None)
    df.columns = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
                  'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
                  'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem',
                  'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay',
                  'mycelium', 'int-discolor', 'slcerotia', 'fruit-pods', 'fruit spots', 'seed',
                  'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots', 'class']
    df['class'] = df['class'].apply(convert_soybean_to_numerical)
    normalized_df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    df = df.sample(frac=1)
    return df

def convert_soybean_to_numerical(x):
    if x == 'D1':
        return 1
    elif x == 'D2':
        return 2
    elif x == 'D3':
        return 3
    elif x == 'D4':
        return 4

def process_voter_data():
    """
    Processes voter data
    :return: DataFrame
    """
    df = pd.read_csv('data/house-votes-84.data', header=None, na_values=['?'])
    df = df.fillna(np.random.randint(0, 1))
    df.columns = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-budget', 'physician-fee-freeze',
                  'el-salvador-aid', 'religious-groups-in-school', 'anti-satellite-test-ban', 'aid-to-contras',
                  'mx-missile', 'immigration', 'synfuels-corp-cutback', 'ed-spending', 'superfunds-right-to-sue',
                  'crime', 'duty-free-exports', 'export-admin-act-sa', 'class']
    df['handicapped-infants'] = df['handicapped-infants'].apply(lambda x: 1 if x == 'y' else 0)
    df['water-project-cost-sharing'] = df['water-project-cost-sharing'].apply(lambda x: 1 if x == 'y' else 0)
    df['adoption-of-budget'] = df['adoption-of-budget'].apply(lambda x: 1 if x == 'y' else 0)
    df['physician-fee-freeze'] = df['physician-fee-freeze'].apply(lambda x: 1 if x == 'y' else 0)
    df['el-salvador-aid'] = df['el-salvador-aid'].apply(lambda x: 1 if x == 'y' else 0)
    df['religious-groups-in-school'] = df['religious-groups-in-school'].apply(lambda x: 1 if x == 'y' else 0)
    df['anti-satellite-test-ban'] = df['anti-satellite-test-ban'].apply(lambda x: 1 if x == 'y' else 0)
    df['aid-to-contras'] = df['aid-to-contras'].apply(lambda x: 1 if x == 'y' else 0)
    df['mx-missile'] = df['mx-missile'].apply(lambda x: 1 if x == 'y' else 0)
    df['immigration'] = df['immigration'].apply(lambda x: 1 if x == 'y' else 0)
    df['synfuels-corp-cutback'] = df['synfuels-corp-cutback'].apply(lambda x: 1 if x == 'y' else 0)
    df['ed-spending'] = df['ed-spending'].apply(lambda x: 1 if x == 'y' else 0)
    df['superfunds-right-to-sue'] = df['superfunds-right-to-sue'].apply(lambda x: 1 if x == 'y' else 0)
    df['crime'] = df['crime'].apply(lambda x: 1 if x == 'y' else 0)
    df['duty-free-exports'] = df['duty-free-exports'].apply(lambda x: 1 if x == 'y' else 0)
    df['export-admin-act-sa'] = df['export-admin-act-sa'].apply(lambda x: 1 if x == 'y' else 0)
    df['class'] = df['class'].apply(lambda x: 1 if x == 'y' else -1)
    return df
