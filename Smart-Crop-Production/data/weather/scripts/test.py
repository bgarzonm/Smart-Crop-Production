import pandas as pd

# Crear el primer dataframe
df1 = pd.DataFrame({'id': [1, 2, 3, 4], 'valor1': [10, 20, 30, 40], 'valor2': [100, 200, 300, 400]})

# Crear el segundo dataframe
df2 = pd.DataFrame({'id': [2, 3, 5, 6], 'valor3': [50, 60, 70, 80], 'valor4': [500, 600, 700, 800]})

# Unir los dataframes basados en la columna 'id'
df_unido = pd.merge(df1, df2, on='id')

# Mostrar el dataframe unido
print(df_unido)
