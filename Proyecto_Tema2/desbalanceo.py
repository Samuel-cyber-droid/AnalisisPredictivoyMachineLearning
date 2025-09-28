import pandas as pd
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('churn_data.csv')

# --- Visualización ANTES del oversampling ---
print("Distribución de clases ANTES del oversampling:")
print(df['Churn'].value_counts())

plt.figure(figsize=(7, 5)) # Define el tamaño de la figura
sns.countplot(x='Churn', data=df)
plt.title('Distribución de Clases ANTES del Oversampling')
plt.xlabel('Cliente Cancela (Churn)')
plt.ylabel('Cantidad de Clientes')
plt.show()


# --- Proceso de Oversampling ---
# Separar las clases
df_majority = df[df['Churn'] == 0]
df_minority = df[df['Churn'] == 1]

# Remuestrear la clase minoritaria
df_minority_oversampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

# Combinar los dataframes
df_oversampled = pd.concat([df_majority, df_minority_oversampled])


# --- Visualización DESPUÉS del oversampling ---
print("\nDistribución de clases DESPUÉS del oversampling:")
print(df_oversampled['Churn'].value_counts())

plt.figure(figsize=(7, 5)) # Define el tamaño de la figura
sns.countplot(x='Churn', data=df_oversampled)
plt.title('Distribución de Clases DESPUÉS del Oversampling')
plt.xlabel('Cliente Cancela (Churn)')
plt.ylabel('Cantidad de Clientes')
plt.show()