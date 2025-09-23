import pandas as pd
from sklearn.prepocessing import MinMaxScaler

df = pd.DataFrame({
    'Tipo_Mascota':['Perro','Gato','Perro','Pez','Gato'],
    'Edad_años': [3,5,8,1,2],
    'Peso_kg':[10,4,25,0.1,3],
    'Jugueton':['Si','No','Si','No','Si']
})

print("--- Dataset Original ---")
print(df)

scaler = MinMaxScaler()
df[['Edad_norm', 'Peso_norm']] = scaler.fit_transform(df[['Edad_años', 'Peso_kg']])

# Eliminamos las columnas numéricas originales
df = df.drop(['Edad_años', 'Peso_kg'], axis=1)

df = pd.get_dummies(df, columns=['Tipo_Mascota', 'Juguetón'], drop_first=True)

print("\n--- Dataset Final Transformado ---")
print(df)