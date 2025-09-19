from sys import prefix

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

df = pd.DataFrame({
    'Edad': [18, 25, 40, 60],
    'Ingresos':[1000, 2500, 4000, 8000],
    'Color':['Rojo', 'Azul', 'Verder', 'Rojo'],
    'Clase':['A', 'B', 'A', 'C']
})

print("Dataset Original:")
print(df)

#Transformaciones Numericas
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()
df[['Edad_norm', 'Ingresos_norm']] = scaler_minmax.fit_transform(df[['Edad', 'Ingresos']])
df[['Edad_std', 'Ingresos_std']] = scaler_std.fit_transform(df[['Edad', 'Ingresos']])

# Transformaciones Categoricas
le = LabelEncoder()
df['Color_le'] = le.fit_transform(df['Color'])
df['Clase_le'] = le.fit_transform(df['Clase'])

# one-hot encoding
df = pd.get_dummies(df, columns=['Color','Clase'], prefix=['Color','Clase'])

print("\n Dataset transformado")
print(df)

pd.set_option('display.max_columns', None)
print(df)