import pandas as pd
dfList=[]
dfNewList=[]
temperatures_RAW = pd.read_csv('./data/data_NEW/temperature_El_Espinal.txt.csv')
dfList.append(temperatures_RAW)
precipitation_RAW=pd.read_csv('./data/data_NEW/precipitation_El_Espinal.txt.csv')
dfList.append(precipitation_RAW)
vapourPressure_RAW=pd.read_csv('./data/data_NEW/vapour_pressure_El_Espinal.txt.csv')
dfList.append(vapourPressure_RAW)
for i in dfList:
    cols=i.columns
    i['Date-temp']=i['Yr'].astype(str) + '-' + i['Mo'].astype(str).str.zfill(2)
    for j in cols:
        if j=='N.Syn':
            temp=i.drop(columns=['Yr','Mo','N.Obs','N.Syn'])
        else:
            temp=i.drop(columns=['Yr','Mo','N.Obs'])
    temp['Date'] = pd.to_datetime(temp['Date-temp'], format='%Y-%m')
    temp.drop('Date-temp', axis=1, inplace=True)
    colsAuxTemp=temp.columns.tolist()
    colsAux=colsAuxTemp[::-1]
    temp=temp.reindex(columns=colsAux)
    dfNewList.append(temp)
oldColumns={'Value(°C)':'Centigrade'}
dfNewList[0]=dfNewList[0].rename(columns=oldColumns)
#print(dfNewList)
final_DF=pd.concat([dfNewList[0].set_index('Date'), dfNewList[1].set_index('Date'), dfNewList[2].set_index('Date')], axis=1)
print(final_DF)#The last and not least dataframe, customized