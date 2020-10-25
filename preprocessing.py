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
