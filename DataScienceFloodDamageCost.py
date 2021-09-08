#!/usr/bin/env python
# coding: utf-8

# In[1]:


#predict the damage costs from floods
import numpy as np
import pandas as pd
df1 = pd.read_csv('Downloads/dataset/average-precipitation-per-year.csv')

grouped_df=df1.groupby(['Year']).sum()
print(grouped_df)


# In[52]:


df2 = pd.read_csv('Downloads/dataset/damage-costs-from-natural-disasters.csv')


# In[53]:


df3 = pd.read_csv('Downloads/dataset/number-injured-from-disasters.csv')
df_merged = pd.merge(df3,df2,on=["Year","Year"])


# In[54]:


df4 = pd.read_csv('Downloads/dataset/natural-disaster-death-rates.csv')
df_merged = pd.merge(df4,df_merged,on=["Year","Year"])
print(df_merged)


# In[55]:


import matplotlib.pyplot as plt

plt.plot(df_merged['Year'],df_merged['Total economic damage (EMDAT (2020))'])
plt.title('Total economic damage in unit dollar')
plt.xlabel('Year')
plt.ylabel('Total economic damage')
plt.show()


# In[5]:


#update the name of columns
df_merged = df_merged.drop(columns=['Code','Entity_x','Code_x','Entity_y', 'Code_y','Entity','Year'])
df_merged = df_merged.rename(columns={'Global death rates from natural disasters' : 'Death Rates'})
df_merged = df_merged.rename(columns={'Injured (EMDAT (2019))':'Number of Injured'})
df_merged = df_merged.rename(columns={'Total economic damage (EMDAT (2020))': 'Total Economic Damage'})

print(df_merged)


# In[7]:


import seaborn as sns

sns.set_style("whitegrid")
sns.pairplot(df_merged,height = 3)


# In[15]:


#Rubbish in rubbish out model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_merged['Total Economic Damage']=df_merged['Total Economic Damage']/10000000000
df_merged['Number of Injured'] = df_merged['Number of Injured']/1000
X =df_merged[['Death Rates','Number of Injured']]
y = np.array(df_merged['Total Economic Damage'])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
lr = LinearRegression().fit(X_train,y_train)
#print(X)
#print(y)

#predict 
y_pred = lr.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(lr.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[17]:


#Cluster then predict technique
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()


df2 = pd.read_csv('Downloads/dataset/damage-costs-from-natural-disasters.csv')
df3 = pd.read_csv('Downloads/dataset/number-injured-from-disasters.csv')
df_merged = pd.merge(df3,df2,on=["Year","Year"])
df4 = pd.read_csv('Downloads/dataset/natural-disaster-death-rates.csv')
df_merged = pd.merge(df4,df_merged,on=["Year","Year"])


df_merged = df_merged.drop(columns=['Code','Entity_x','Code_x','Entity_y', 'Code_y','Entity','Year'])

#data normalization
df_merged['Total economic damage (EMDAT (2020))']=df_merged['Total economic damage (EMDAT (2020))']/10000000

df_merged = df_merged.rename(columns={'Global death rates from natural disasters' : 'Death Rates'})
df_merged = df_merged.rename(columns={'Injured (EMDAT (2019))':'Number of Injured'})
df_merged = df_merged.rename(columns={'Total economic damage (EMDAT (2020))': 'Total Economic Damage(x10000000)'}) 
#(x10000000) is put on the column name so that we remember what we have done on that column



X =df_merged[['Death Rates','Number of Injured']]
y = np.array(df_merged['Total Economic Damage(x10000000)'])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)

kmeans = KMeans(n_clusters=3)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

pipeline.fit(df_merged)
print("cluster center")
print(kmeans.cluster_centers_)
y = kmeans.labels_ 
#print(y[5])


cluster_1 = np.empty((0,3))
cluster_2= np.empty((0,3))
cluster_3= np.empty((0,3))
for i in range(3):
    count=0
    for j in range(len(X_train)):
        if y[j]==0:
            cluster_1 = np.append(cluster_1,np.array([df_merged.loc[j, :]]), axis=0)  # Subset of the datapoints that have been assigned to the cluster i
        elif y[j]==1:
            cluster_2 = np.append(cluster_2,np.array([df_merged.loc[j, :]]),axis=0)
        elif y[j]==2:
            cluster_3 = np.append(cluster_3,np.array([df_merged.loc[j, :]]),axis=0)
#   Do analysis on this subset of datapoints.
print("cluster points")
print(cluster_1)
print(cluster_2)
print(cluster_3)


# In[18]:



#Cluster is needed because not all number-injured-from-floods and natural-disaster-death-floods  can related with 
#damage-costs-from-floods. The number of injured and death rates cannot determine the damage-costs-from-floods 
#because maybe the floods occur on place with less dense population but have high cost of property like industry.
#If the floods occur on place with dense population, usually denser population means the place have many properties own
#by them and it will be higher cost damage when floods occur. 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

X = cluster_1[:,0:2]
#print(X)
y=cluster_1[:,2]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
lr = LinearRegression()

pipeline = make_pipeline(normalizer,lr)

pipeline.fit(X_train,y_train)

#predict 
y_pred = pipeline.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(pipeline.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

X = cluster_2[:,0:2]
#print(X)
y=cluster_2[:,2]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
lr = LinearRegression()

pipeline = make_pipeline(normalizer,lr)

pipeline.fit(X_train,y_train)

#predict 
y_pred = pipeline.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(pipeline.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

X = cluster_3[:,0:2]
#print(X)
y=cluster_3[:,2]
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size =0.2, random_state=42)
lr = LinearRegression()

pipeline = make_pipeline(normalizer,lr)

pipeline.fit(X_train,y_train)

#predict 
y_pred = pipeline.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(pipeline.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[39]:


import pandas as pd
df5 = pd.read_csv('Downloads/dataset/bantuanbencanamengikutjenis2015.csv')

print(df5)
df5 = df5.drop([0,1,2,3,10])
df5 = df5.rename(columns={'BANTUAN BENCANA MENGIKUT JENIS BENCANA,  2015' : 'Jenis Bencana / Type of Disaster'})
df5 = df5.rename(columns={'Unnamed: 1':'Bil. Penerima/No. of recipient'})
df5 = df5.rename(columns={'Unnamed: 2':'Bantuan/Helps (RM)'})

print(df5)


# In[45]:


#df5= df5.to_numpy
df5['Bil. Penerima/No. of recipient'] = pd.to_numeric(df5['Bil. Penerima/No. of recipient'])
df5['Bantuan/Helps (RM)'] = pd.to_numeric(df5['Bantuan/Helps (RM)'])
np5_1 = np.array(df5['Jenis Bencana / Type of Disaster'])
np5_2 = np.array(df5['Bil. Penerima/No. of recipient'])
np5_3 =np.array(df5['Bantuan/Helps (RM)'])


# In[51]:


from matplotlib import pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
plt.title('No. of recipient in the financial helps')
langs = np5_1
ax.pie(np5_2, labels = langs,autopct='%1.2f%%')
plt.show()


# In[50]:



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
plt.title('Amount of money in the helps (RM)')
langs = np5_1
ax.pie(np5_3, labels = langs,autopct='%1.2f%%')
plt.show()

