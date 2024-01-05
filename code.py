import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('segmentation data.csv')

df.drop(['ID'], inplace = True, axis = 1)

describe_num_df = pd.DataFrame()
describe_num_df['Age'] = df['Age'].describe()
describe_num_df['Income'] = df['Income'].describe()

col_names = df.columns
features = df[col_names]

scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled = pd.DataFrame(features, columns = col_names)

data=scaled[['Age','Income']]

kmeans=KMeans(n_clusters=4,random_state=0) 
kmeans.fit(data)

prediction=kmeans.fit_predict(data)
prediction

clustered_data = df.copy()
clustered_data["cluster_index"] = prediction

sns.scatterplot(x=clustered_data.Age,
                y=clustered_data.Income,
                hue=clustered_data.cluster_index,
                palette="deep")
plt.show()
