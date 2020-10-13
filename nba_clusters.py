#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go


# In[ ]:


df=pd.read_csv(r'C:\Users\st-am\OneDrive\Documents\Data analytics\nba_shooting\nba_shooting.csv',sep=';',               skipinitialspace=(True))


# ### Data cleaning

# In[ ]:


#Remove unwanted -
for i in range(4,29):
    df=df[df.iloc[:,i] != '#VALEUR!']

df = df.reset_index(drop=True)
    
#Fix height value
for i in range(len(df)):
    df['HEIGHT'][i] = df['HEIGHT'][i].replace("'", ".")

#Change data type to float
df[df.columns[2:29]] = df[df.columns[2:29]].astype(float)


#Select players with at least 15min per game
df=df[df.MINUTES > 12]
df = df.reset_index(drop=True)


# ### Feature creation 

# In[ ]:


#Add some variables (for future analysis)

shot_attempts=df.filter(regex='FGA').sum(axis=1)
df['SHOTS']=shot_attempts

shots_perimeter = df.filter(regex='3_FGA').sum(axis=1)


# In[ ]:


df['CLOSE%']=df['PAINT_FGA']/df['SHOTS']
df['MID_RANGE%']=df['MR_FGA']/df['SHOTS']
df['PERIMETER%']=shots_perimeter/df['SHOTS']
df['REBOUNDS']=df['OREB']+df['DREB']


# In[ ]:


df.shape


# In[ ]:


df.head()


# ### Player clustering

# In[ ]:


data = df.iloc[:,5:34]
scaled_data = MinMaxScaler().fit_transform(data)


# In[ ]:


#Find best number of clusters
sil = []
kmax = 12
my_range=range(6,kmax+1)

for i in my_range:
    kmeans = KMeans(n_clusters = i).fit(scaled_data)
    labels = kmeans.labels_
    sil.append(silhouette_score(scaled_data, labels, metric = 'euclidean'))


# In[ ]:


#Plot it
plt.plot(my_range, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by K')
plt.show()


# In[ ]:


#Proceed with best number of clusters
algo = KMeans(n_clusters=7, random_state=123)
algo.fit(scaled_data)

df['LABELS']=algo.labels_


# ### ANALYSIS

# In[ ]:


#Analyze labels
clusters=pd.DataFrame(columns=(df.iloc[:,2:34]).columns)

for i in range(0,7):
    a=df[df['LABELS']==i].mean()
    clusters=clusters.append(a, ignore_index=True)


# In[ ]:


df[df['LABELS']==6]


# In[ ]:


cluster_comparison = clusters[['LABELS','MINUTES','PPG','AST','REBOUNDS','CLOSE%','MID_RANGE%','PERIMETER%',
                              'DRIVES','POSTUPS','CATCH_SHOOT']].round(2)
cluster_comparison


# ### PLOTS

# In[ ]:


good_teams=['LAL','LAC','MIL','BOS','TOR','HOU','MIA','DEN']

a=df[df['TEAM'].isin(good_teams)]
b=df[~df['TEAM'].isin(good_teams)]


# In[ ]:


#Good teams computations
test=pd.DataFrame(a['LABELS'].value_counts().sort_index()).transpose().reset_index(drop=True)
test=test/8
r=test.iloc[0]


# In[ ]:


test.columns = ['Cluster' + str(col) for col in test.columns]
cat1 = ['Stretch Players', 'High Usage Bigs', 'Low Usage Bigs', 'Ball Dominant Scorers', 'Versatile Rotation Players',
       'High Quality Contributors', 'Athletic Forwards']


# In[ ]:


#Rest of the league
test2=pd.DataFrame(b['LABELS'].value_counts().sort_index()).transpose().reset_index(drop=True)
test2=test2/22
r2=test2.iloc[0]


# In[ ]:


#Radar plot
fig=go.Figure()
fig.add_trace(go.Scatterpolar(r=r, theta=cat1,fill='toself',name='Good Teams'))
fig.add_trace(go.Scatterpolar(r=r2, theta=cat1,fill='toself',name='Other Teams'))
fig.show()


# In[ ]:




