# Import necessary libraries
from sklearn import datasets # to retrieve the iris Dataset
import pandas as pd # to load the dataframe
from sklearn.preprocessing import StandardScaler # to standardize the features
from sklearn.decomposition import PCA # to apply PCA
import seaborn as sns # to plot the heat maps
import matplotlib.pyplot as plt
#Load the Dataset
iris = datasets.load_iris ()
df = pd.DataFrame(iris['data'], columns = iris['feature_names'])
print(df.head ())
scalar = StandardScaler()
scaled_data = pd.DataFrame (scalar.fit_transform (df)) 
sns.heatmap(scaled_data.corr())
plt.show()
pca = PCA(n_components = 3)
pca.fit(scaled_data)
data_pca = pca.transform(scaled_data)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3'])
data_pca.head()
sns.heatmap(data_pca.corr())
plt.show()