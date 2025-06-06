# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 18:02:39 2025

@author: andre
"""
#Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2 cargar datos
df = pd.read_csv('C:/Users/andre/Documents/Python Scripts/titanic/test.csv')  # Ajusta la ruta del archivo
print(df.head())  # Ver primeras filas
#3 exploracion inicial
df.info()  # Tipos de datos y valores nulos
df.describe()  # Estadísticas descriptivas

#4 eliminar columnas irrelevantes.
df = df.drop(['PassengerId', 'Cabin'], axis=1)  # Ejemplo: eliminar columnas no útiles
#manejo de valores nulos
df = df.dropna(subset=['Embarked'])  # Solo si son pocos registros
#imputar valores
df['Age'].fillna(df['Age'].median(), inplace=True)  # Llenar con mediana
df['Fare'].fillna(df['Fare'].mean(), inplace=True)  # Llenar con promedio
#elimina duplicados
df = df.drop_duplicates()
#corregir formatos y tipos de datos
df['Sex'] = df['Sex'].astype('category')  # Convertir a categoría
df['Name'] = df['Name'].str.title()  # Capitalizar nombres
#manejo de Outliers
# Usar rango intercuartílico (IQR) para detectar outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]

#5 validacion y exportación
df.info()  # Confirmar que no hay nulos
df.head()  # Ver estructura final

df.to_csv('C:/Users/andre/Documents/Python Scripts/titanic/titanic_clean.csv', index=False)  # Guardar como CSV

#6 visualización
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribución de Edades')
plt.show()