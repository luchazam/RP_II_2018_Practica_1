import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['buying','maint','doors','persons','lug_boot','safety','target'])

from sklearn.preprocessing import StandardScaler
features = ['buying','maint','doors','persons','lug_boot','safety']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 6)

#vec = DictVectorizer()
#x = vec.fit_transform(x)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dist_le = LabelEncoder()

for i in range(0,x.shape[1]):
	x[:,i] = dist_le.fit_transform(x[:,i])

ohe = OneHotEncoder(categorical_features='all')
ohe.fit_transform(x)
x=ohe.transform(x).toarray()
x = StandardScaler().fit_transform(x)

n_atributes = x.shape[1]

pca = PCA(n_components=n_atributes)
principalComponents = pca.fit_transform(x)

componentes = range(principalComponents.shape[1])

principalDf = pd.DataFrame(data = principalComponents, columns = componentes)
print(pca.explained_variance_ratio_)
plt.figure(1)
x = StandardScaler().fit_transform(x)
plt.bar(range(n_atributes),pca.explained_variance_ratio_)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['unacc', 'acc', 'good', 'vgood']
colors = ['r', 'g', 'b', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
