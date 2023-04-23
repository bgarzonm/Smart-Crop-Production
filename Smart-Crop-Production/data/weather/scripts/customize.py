import os
directoryFiles = "./data/data_RAW"
newDirectoryFiles="./data/data_NEW"
typeFile = ".txt"
for file in os.listdir(directoryFiles):
    new_Data=[]
    try:
        if file.endswith(typeFile):
            with open(f"{directoryFiles}/{file}", "r") as f_old:
                for i in f_old:
                    i=' '.join(i.split())
                    for j in range(len(i)-1):
                        if i[j] == ' ':
                            i_lista = list(i)
                            i_lista[j] = ","
                            i = ''.join(i_lista)
                    new_Data.append(i)
            data_Final='\n'.join(new_Data)
            with open(f"{newDirectoryFiles}/{file}.csv", 'w') as f_new:
                f_new.write(data_Final)
    except:
        pass