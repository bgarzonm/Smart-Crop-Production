import pandas as pd

# Crear algunos DataFrames de ejemplo
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D', 'E'], 'value2': [5, 6, 7, 8]})

# Utilizar la función join()
df_join = df1.join(df2.set_index('key'), on='key')
print(df_join)

# Utilizar la función concat()
df_concat = pd.concat([df1.set_index('key'), df2.set_index('key')], axis=1)
print(df_concat)
