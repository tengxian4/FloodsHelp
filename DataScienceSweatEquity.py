#!/usr/bin/env python
# coding: utf-8

# In[2]:


#floods data from Malaysia only, predict how many people affected during floods
import numpy as np
import pandas as pd
df1 = pd.read_excel('Downloads/dataset/Dataset/JumlahPerpindahanMangsaBanjir(orang).xlsx')
print(df1)

df1.columns = df1.columns.map(str)

#impute missing value with mean
for col in range(1,len(df1.columns)):
    count =0
    sum=0
    for row in range(0,15):
        if df1.iloc[row,col] != 0:
            count = count +1
            sum =sum + df1.iloc[row,col]
    mean = sum/count
    for row in range(0,15):
        if df1.iloc[row,col] == 0:
            df1.iloc[row,col]=mean
            
print(df1)


# In[3]:


import numpy as np
import pandas as pd
df2 = pd.read_excel('Downloads/dataset/Dataset/Kedalaman Banjir Maksimum(m).xlsx')
print(df2)

df2.columns = df2.columns.map(str)

#impute missing value with mean
for col in range(1,len(df2.columns)):
    count =0
    sum=0
    for row in range(0,15):
        if df2.iloc[row,col] != 0:
            count = count +1
            sum =sum + df2.iloc[row,col]
    mean = sum/count
    for row in range(0,15):
        if df2.iloc[row,col] == 0:
            df2.iloc[row,col]=mean
            
print(df2)


# In[4]:


import numpy as np
import pandas as pd
df3 = pd.read_excel('Downloads/dataset/Dataset/PurataHujanHarianTertinggi(mm).xlsx')
print(df3)

df3.columns = df3.columns.map(str)

#impute missing value with mean
for col in range(1,len(df3.columns)):
    count =0
    sum=0
    for row in range(0,15):
        if df3.iloc[row,col] != 0:
            count = count +1
            sum =sum + df3.iloc[row,col]
    mean = sum/count
    for row in range(0,15):
        if df3.iloc[row,col] == 0:
            df3.iloc[row,col]=mean
            
print(df3)


# In[5]:


#change the format of the table
mangsa = pd.DataFrame(columns=["Negeri_Year","JumlahPerpindahanMangsaBanjir(orang)"])
colName = df1.columns
negeri = df1.Negeri
for col in range(1,len(df1.columns)):
    for row in range(0,15):
        mangsa=mangsa.append({'Negeri_Year': negeri[row]+colName[col] ,'JumlahPerpindahanMangsaBanjir(orang)':df1.iloc[row,col]},ignore_index=True)  
print(mangsa)


# In[8]:


#change the format of the table
KedalamanMaksimum = pd.DataFrame(columns=["Negeri_Year","Kedalaman Banjir Maksimum(m)"])
colName = df2.columns
negeri = df2.Negeri
for col in range(1,len(df2.columns)):
    for row in range(0,15):
        KedalamanMaksimum = KedalamanMaksimum.append({'Negeri_Year': negeri[row]+colName[col] ,'Kedalaman Banjir Maksimum(m)':df2.iloc[row,col]},ignore_index=True)  
print(KedalamanMaksimum)


# In[6]:


#change the format of table
purataHujan = pd.DataFrame(columns=["Negeri_Year","PurataHujanHarianTertinggi(mm)"])
colName = df3.columns
negeri = df3.Negeri
for col in range(1,len(df3.columns)):
    for row in range(0,15):
        purataHujan = purataHujan.append({'Negeri_Year': negeri[row]+colName[col] ,'PurataHujanHarianTertinggi(mm)':df3.iloc[row,col]},ignore_index=True)  
print(purataHujan)


# In[12]:


#merge the data with similar state and year
df_merged = pd.merge(KedalamanMaksimum,purataHujan,on=["Negeri_Year","Negeri_Year"])
df_merged = pd.merge(mangsa,df_merged,on=["Negeri_Year","Negeri_Year"])

print(df_merged)

#change the unit with divide 100000
#QUITE BIG floating number similar with other column normalization of data
df_merged['JumlahPerpindahanMangsaBanjir(orang)'] =df_merged['JumlahPerpindahanMangsaBanjir(orang)']/100000


# In[10]:


import seaborn as sns
df_merged['JumlahPerpindahanMangsaBanjir(orang)'] =df_merged['JumlahPerpindahanMangsaBanjir(orang)']*100000
graph = df_merged[['Kedalaman Banjir Maksimum(m)','PurataHujanHarianTertinggi(mm)','JumlahPerpindahanMangsaBanjir(orang)']]
sns.set_style("whitegrid")
sns.pairplot(graph,height = 3)
df_merged['JumlahPerpindahanMangsaBanjir(orang)'] =df_merged['JumlahPerpindahanMangsaBanjir(orang)']/100000


# In[19]:


import matplotlib.pyplot as plt
x =df_merged['PurataHujanHarianTertinggi(mm)']
y = df_merged['Kedalaman Banjir Maksimum(m)']
z = df_merged['JumlahPerpindahanMangsaBanjir(orang)']
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter(x, y, z)

plt.xlabel('PurataHujanHarianTertinggi(mm)')
plt.ylabel('Kedalaman Banjir Maksimum(m)')
ax.set_zlabel('JumlahPerpindahanMangsaBanjir(orang)')

plt.show()
df_merged['JumlahPerpindahanMangsaBanjir(orang)'] =df_merged['JumlahPerpindahanMangsaBanjir(orang)']/100000


# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X = df_merged[['Kedalaman Banjir Maksimum(m)','PurataHujanHarianTertinggi(mm)']]
y = df_merged['JumlahPerpindahanMangsaBanjir(orang)']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
lr = LinearRegression().fit(X_train,y_train)

#predict 
y_pred = lr.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(lr.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[9]:





# In[14]:


#after translation for viewing purpose
df_new = df_merged.rename(columns={ 'Kedalaman Banjir Maksimum(m)':'Maximum Flood Depth(m)'})
df_new = df_new.rename(columns={ 'PurataHujanHarianTertinggi(mm)':'AverageHighestDailyRainfall(mm)'})
df_new = df_new.rename(columns={'JumlahPerpindahanMangsaBanjir(orang)':'Number of Evacuated Flood Victims '})
df_new = df_new.rename(columns={'Negeri_Year':'State_Year'})

print(df_new)


# In[ ]:


sns.pairplot

