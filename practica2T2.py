import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Crear el DataFrame original (empezamos de cero)
df = pd.DataFrame({
    'Tipo_Mascota': ['Perro', 'Gato', 'Perro', 'Pez', 'Gato'],
    'Edad_años': [3, 5, 8, 1, 2],
    'Peso_kg': [10, 4, 25, 0.1, 3],
    'Juguetón': ['Si', 'No', 'Si', 'No', 'Si']
})

# Guardamos una copia para comparar al final
df_original = df.copy()

# 2. Transformaciones numéricas
scaler = MinMaxScaler()
df[['Edad_años', 'Peso_kg']] = scaler.fit_transform(df[['Edad_años', 'Peso_kg']])

# 3. Transformaciones categóricas (One-Hot Encoding)
# Se aplica a las columnas de texto y las reemplaza
df = pd.get_dummies(df, columns=['Tipo_Mascota', 'Juguetón'], drop_first=True)

print("--- Dataset Original ---")
print(df_original)
print("\n--- Dataset Final Transformado ---")
print(df)