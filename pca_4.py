import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

url = 'yeast.data'
# load dataset into Pandas DataFrame
df = pd.read_csv(url,names=['Sequence Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','target'])

from sklearn.preprocessing import StandardScaler
features = ['mcg','gvh','alm','mit','erl','pox','vac','nuc']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

n_atributes = x.shape[1]

pca = PCA(n_components=n_atributes)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5','principal component 6','principal component 7','principal component 8'])
print(pca.explained_variance_ratio_)
plt.figure(1)
plt.bar(range(n_atributes),pca.explained_variance_ratio_)
#plt.show()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

plt.figure(2)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['CYT','NUC','MIT','ME3','ME2','ME1','EXC','VAC','POX','ERL']
colors = ['r','g','b','k','tomato','y','violet','c','olive','indigo']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
