import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df=pd.read_excel(r"C:/Users/ADMIN/Downloads/Blog_TFIDF_Vectors.xlsx")

# Q1
def equal_width_binning(data,bins=4):
    data=np.array(data)
    mn=data.min()
    mx=data.max()
    width=(mx-mn)/bins
    edges=[mn+i*width for i in range(bins+1)]
    binned=np.digitize(data,edges[1:-1],right=False)
    return binned

def entropy(values):
    values=np.array(values)
    unique,counts=np.unique(values,return_counts=True)
    p=counts/len(values)
    return -np.sum(p*np.log2(p))

# Q2
def gini_index(values):
    values=np.array(values)
    unique,counts=np.unique(values,return_counts=True)
    p=counts/len(values)
    return 1-np.sum(p**2)

# Q3
def information_gain(data,feature,target):
    total_entropy=entropy(data[target])
    vals,counts=np.unique(data[feature],return_counts=True)
    weighted_entropy=0
    for v,c in zip(vals,counts):
        subset=data[data[feature]==v]
        weighted_entropy+=(c/len(data))*entropy(subset[target])
    return total_entropy-weighted_entropy

def root_node(data,target):
    features=[col for col in data.columns if col!=target]
    gains=[information_gain(data,f,target) for f in features]
    return features[np.argmax(gains)]

# Q4
def equal_freq_binning(data,bins=4):
    data=np.array(data)
    quantiles=np.percentile(data,np.linspace(0,100,bins+1))
    binned=np.digitize(data,quantiles[1:-1],right=False)
    return binned

def bin_feature(data,method="width",bins=4):
    if method=="width":
        return equal_width_binning(data,bins)
    if method=="frequency":
        return equal_freq_binning(data,bins)

# Q5
class Node:
    def __init__(self,feature=None,label=None):
        self.feature=feature
        self.label=label
        self.children={}

def build_tree(data,target):
    if len(np.unique(data[target]))==1:
        return Node(label=data[target].iloc[0])
    features=[col for col in data.columns if col!=target]
    if len(features)==0:
        return Node(label=data[target].mode()[0])
    gains=[information_gain(data,f,target) for f in features]
    best=features[np.argmax(gains)]
    node=Node(feature=best)
    for v in np.unique(data[best]):
        subset=data[data[best]==v].drop(columns=[best])
        node.children[v]=build_tree(subset,target)
    return node

# Q6
def visualize_tree(X,y):
    clf=DecisionTreeClassifier()
    clf.fit(X,y)
    plt.figure(figsize=(10,6))
    plot_tree(clf,feature_names=X.columns,class_names=True,filled=True)
    plt.show()

# Q7
def plot_decision_boundary(X,y):
    model=DecisionTreeClassifier()
    model.fit(X,y)
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
    Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,alpha=0.4)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.show()

target=df.columns[-1]

for col in df.columns:
    if df[col].dtype!="object" and col!=target:
        df[col]=bin_feature(df[col],"width",4)

print(entropy(df[target]))
print(gini_index(df[target]))

root=root_node(df,target)
print(root)

tree_model=build_tree(df,target)

X=df.drop(columns=[target])
y=df[target]

visualize_tree(X,y)

X2=X.iloc[:,0:2].values
plot_decision_boundary(X2,y)