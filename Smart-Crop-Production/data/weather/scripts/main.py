import pandas as pd

# Leer el archivo como un dataframe
temperatures_RAW = pd.read_csv('temperature_El_Espinal.txt', delimiter='\t')
precipitation_RAW=pd.read_csv('precipitation_El_Espinal.txt', delimiter='\t')
vapourPressure_RAW=pd.read_csv('vapour_pressure_El_Espinal.txt', delimiter='\t')
print(temperatures_RAW.columns)
# Mostrar el dataframe
#print(temperatures)
temperatures=temperatures_RAW.drop('N.Obs', axis=1)
precipitation=precipitation_RAW.drop(columns=['N.Obs'])
vapourPressure=vapourPressure_RAW.drop(columns=['N.Obs', 'N.Syn'])
temperatures['fecha-temp']=temperatures['Yr'].astype(str) + '-' + temperatures['Mo'].astype(str).str.zfill(2)
print(temperatures)
#print(temperatures.columns, precipitation.columns, vapourPressure.columns)
df_concatenado = pd.concat([temperatures, precipitation, vapourPressure])
#print(df_concatenado)