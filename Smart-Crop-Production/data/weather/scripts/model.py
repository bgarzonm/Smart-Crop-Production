#manejo de datos
import numpy as np
import pandas as pd
#manejo de visual
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import hvplot.pandas
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import plotly.express as px
import plotly.graph_objects as go
#modelacion arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller #mirar si es estacionaria
#modelo auto-arima
from pmdarima import auto_arima
#métrica de evaluación
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from sklearn import metrics
#prevención de advertencias
import warnings
#demas librerias
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth']=1.5
dark_Style={
    'figure.facecolor' : '#212946',
    'axes.facecolor' : '#212946',
    'savefig.facecolor' : '#212946',
    'axes.grid' : True,
    'axes.grid.which' : 'both',
    'axes.spines.left' : False,
    'axes.spines.right' : False,
    'axes.spines.top' : False,
    'axes.spines.bottom' : False,
    'grid.color' : '#2A3459',
    'grid.linewidth' : '1',
    'text.color' : '0.9',
    'axes.labelcolor' : '0.9',
    'xtick.color' : '0.9',
    'ytick.color' : '0.9',
    'font.size' : 12}
plt.rcParams.update(dark_Style)

def evaluacion_modelo(y_true,y_pred):
    def mean_absolute_percentage_error(y_true,y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true-y_pred)/y_true))*100
    print("Evaluación del modelo: ")
    print(f'MSE is : {metrics.mean_squared_error(y_true,y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true,y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true,y_pred))}')
    print(f'MAPE is : {metrics.mean_absolute_percentage_error(y_true,y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true,y_pred)}',end='\n\n')
#Se van a utilizar 4 modelos, ARIMA (media móvil integrada autorregresiva), LSTM (red neuronal de memoria a largo plazo), Random Forest y Facebook Prophet
df = pd.read_csv('data/data_NEW/finalData.csv')
#print(df.head)
#empezamos el analisis exploratorio de datos
#print(df.info())
#convertir a un formato datetime
df['Date']=pd.to_datetime(df['Date'])
#print(df.info())
#ahora vamos a indexar el tiempo
df=df.set_index('Date')
df.index.freq='MS'
#print(df.head())
#print(df.info())
'''figTemperature = px.line(df, x=df.index, y='Centigrade', template="plotly_dark", title="Temperatura en el Espinal")
figTemperature.show()
figPrecipitation = px.line(df, x=df.index, y='Value(mm/month)', template="plotly_dark", title="Precipitación en el Espinal")
figPrecipitation.show()
figPreassure = px.line(df, x=df.index, y='Value(HPa)', template="plotly_dark", title="Presión en el Espinal")
figPreassure.show()'''
#Mostrar densidad de datos y resumen estadistico de cada variable
'''print("Temperatura en El Espinal")
df['Centigrade'].plot(kind='kde',figsize=(16,5))
plt.show()
print(df['Centigrade'].describe())

print("Precipitación en El Espinal")
df['Value(mm/month)'].plot(kind='kde',figsize=(16,5))
plt.show()
print(df['Value(mm/month)'].describe())

print("Presión en El Espinal")
df['Value(HPa)'].plot(kind='kde',figsize=(16,5))
plt.show()
print(df['Value(HPa)'].describe())
'''
#Grafico bloxplot para estacionalidad anual
'''datos=df.copy()
figTemperatura, ax = plt.subplots(figsize=(10,3))
datos['mes']=datos.index.month
datos.boxplot(column='Centigrade',by='mes',ax=ax,color='red')
datos.groupby('mes')['Centigrade'].median().plot(style='o-',linewidth=0.8,ax=ax)
ax.set_ylabel('Temperatura (°C)')
ax.set_title('Distribución de la temperatura en El Espinal')
figTemperatura.suptitle(' ')
plt.show()

figPrecipitacion, ax = plt.subplots(figsize=(10,3))
datos['mes']=datos.index.month
datos.boxplot(column='Value(mm/month)',by='mes',ax=ax,color='red')
datos.groupby('mes')['Value(mm/month)'].median().plot(style='o-',linewidth=0.8,ax=ax)
ax.set_ylabel('Precipitación (mm)')
ax.set_title('Distribución de la precipitación en El Espinal')
figPrecipitacion.suptitle(' ')
plt.show()

figPresion, ax = plt.subplots(figsize=(10,3))
datos['mes']=datos.index.month
datos.boxplot(column='Value(HPa)',by='mes',ax=ax,color='red')
datos.groupby('mes')['Value(HPa)'].median().plot(style='o-',linewidth=0.8,ax=ax)
ax.set_ylabel('Presión (HPa)')
ax.set_title('Distribución de presion en El Espinal')
figPresion.suptitle(' ')
plt.show()'''
#prueba para verificar si los datos son estacionarios
def Prueba_Dickey_Fuller(series , column_name):
    print (f'Resultados de la prueba de Dickey-Fuller para columna: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Número de observaciones utilizadas'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:====>")
        print("Rechazar la hipótesis nula")
        print("Los datos son estacionarios")
    else:
        print("Conclusion:====>")
        print("No se puede rechazar la hipótesis nula")
        print("Los datos no son estacionarios")
#Prueba_Dickey_Fuller(df['Centigrade'],'Centigrade')No es estacionario
#Prueba_Dickey_Fuller(df['Value(mm/month)'],'Value(mm/month)') es estacionario
#Prueba_Dickey_Fuller(df['Value(HPa)'],'Value(HPa)')No es estacionario
#Necesitamos estacionalizar los datos
dfE=df.copy()
#Take first difference
dfE['Temperatura_diff']=df['Centigrade'].diff()
#Removiendo los datos nulos
dfE.dropna(inplace=True)
#print(dfE.head())
#Prueba_Dickey_Fuller(dfE['Temperatura_diff'],'Temperatura_diff') Los datos son estacionarios
dfEP=df.copy()
dfEP['Presión_diff']=df['Value(HPa)'].diff()
dfEP.dropna(inplace=True)
#Prueba_Dickey_Fuller(dfEP['Presión_diff'],'Presión_diff') Los datos son estacionarios
#mostramos datos
'''fig=px.line(dfE, x=dfE.index, y='Temperatura_diff', template='plotly_dark',title='Temperatura en El Espinal')
fig.show()
fig2=px.line(dfEP, x=dfEP.index, y='Presión_diff', template='plotly_dark',title='Presión en El Espinal')
fig2.show()'''
#grafico de autocorrelación temperatura
'''fig, ax= plt.subplots(figsize=(7,3))
plot_acf(dfE['Temperatura_diff'],ax=ax,lags=60,color='white')
plt.show()
fig, ax= plt.subplots(figsize=(7,3))
plot_pacf(dfE['Temperatura_diff'],ax=ax,lags=60,color='white')
plt.show()'''
#grafico de autocorrelación precipitacion
'''fig, ax= plt.subplots(figsize=(7,3))
plot_acf(df['Value(mm/month)'],ax=ax,lags=60,color='white')
plt.show()
fig, ax= plt.subplots(figsize=(7,3))
plot_pacf(df['Value(mm/month)'],ax=ax,lags=60,color='white')
plt.show()'''
#grafico de autocorrelación precipitacion
'''fig, ax= plt.subplots(figsize=(7,3))
plot_acf(dfEP['Presión_diff'],ax=ax,lags=60,color='white')
plt.show()
fig, ax= plt.subplots(figsize=(7,3))
plot_pacf(dfEP['Presión_diff'],ax=ax,lags=60,color='white')
plt.show()'''
#descomposición de la serie de tiempo
'''
plt.rcParams['figure.figsize']=(12,8)
a = seasonal_decompose(df['Centigrade'], model="add")
a.plot()
plt.show()
plt.rcParams['figure.figsize']=(12,8)
a = seasonal_decompose(df['Value(mm/month)'], model="add")
a.plot()
plt.show()
plt.rcParams['figure.figsize']=(12,8)
a = seasonal_decompose(df['Value(HPa)'], model="add")
a.plot()
plt.show()'''
#División de para entrenamiento y prueba
dfTemperatura=df.drop(['Value(HPa)', 'Value(mm/month)'], axis=1)
train_dataTemperatura=dfTemperatura[:len(dfTemperatura)-16]
test_data=dfTemperatura[len(dfTemperatura)-16:]
test=test_data.copy()
print(train_dataTemperatura)
#print(test_data.shape)
#print(train_data,test_data)
#modelo auto-arimam para obtener los mejores parametros p,d,q,P,D,Q(mayuscula es la parte estacionaria)
'''modelo_auto=auto_arima(train_dataTemperatura,start_p=0,d=1,start_q=0,
          max_p=4,max_d=2,max_q=4, start_P=0,
          D=1, start_Q=0, max_P=2,max_D=1,
          max_Q=2, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)
#print(modelo_auto)
#print(modelo_auto.summary())'''
arima_model = SARIMAX(train_dataTemperatura["Centigrade"], order = (2,1,1), seasonal_order = (2,1,0,12)) 
arima_result = arima_model.fit() 
arima_result.summary()
# Gráfico de línea de errores residuales
#residuals = pd.DataFrame(arima_result.resid)
#residuals.plot(figsize = (16,5))
#plt.show()
#diagrama de densidad del jernel de errores residuales
#residuals.plot(kind='kde',figsize=(16,5))
#plt.show()
#print(residuals.describe())
#Entender los resultados
#plt.style.use('seaborn')
#modelo_auto.plot_diagnostics(figsize=(20,8))
#plt.show()
#predicción con el modelo
arima_pred=arima_result.predict(start=len(train_dataTemperatura),end=len(dfTemperatura)-1,typ="levels").rename("ARIMA Predictions")
#print(arima_pred)
arima_pred2 = arima_result.predict(start='2018-01-01',end='2025-01-01', typ="levels").rename("ARIMA Predictions")
print(arima_pred2)

#plt.style.use('dark_background')
plt.rcParams["figure.figsize"] = (20, 8)

plt.plot(test_data["Centigrade"], label="Temperatura actual")
plt.plot(arima_pred, color="lime", label="Predicciones")
plt.title("Predicción con Modelo Arima", fontsize=30)
plt.xlabel('Meses')
plt.ylabel('')
plt.legend( fontsize=16)
plt.show()

plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (20, 8)

plt.plot(test_data["Centigrade"],color="blue" ,label="Temperatural")
plt.plot(arima_pred2, color="lime", label="Predicciones")
plt.title("Predicción con Modelo Arima", fontsize=30)
plt.xlabel('Meses')
plt.ylabel('')
plt.legend( fontsize=16)
plt.show()
evaluacion_modelo(test_data,arima_pred)
test_data['ARIMA_Predictions'] = arima_pred
print(test_data)