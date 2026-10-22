import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset('titanic')

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df.drop(columns=['deck'], inplace=True)

Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['fare'] = np.where(df['fare'] > upper_bound, upper_bound, df['fare'])
df['fare'] = np.where(df['fare'] < lower_bound, lower_bound, df['fare'])

df.drop(columns=['alive', 'class', 'who', 'adult_male'], inplace=True)
df = pd.get_dummies(df, columns=['sex', 'embarked', 'embark_town', 'alone'], drop_first=True)
