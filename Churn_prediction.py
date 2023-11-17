import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os

data = pd.read_csv('/content/drive/MyDrive/task1.csv')
data

df = data.copy()
df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)
df.head()

# checking for null values
overview = pd.concat((df.isnull().sum(),df.nunique(),df.dtypes), axis=1, keys=['Null count','cardinality','Data Type'])
display(overview.T)

# checking correlation
df.corrwith(df.Exited).drop(columns = ['Exited']).sort_values(ascending=False)*100

# better way of same representation
plt.figure(figsize=(5,5))
sns.heatmap(df.corr(), cbar=False, annot=True, fmt='.2f')

# label encoding for some categorical features
gender_labels = {'Female':0, 'Male':1}
geography_labels = {'France':0, 'Germany':1, 'Spain':2}
df.Gender = df.Gender.map(gender_labels)
df.Geography = df.Geography.map(geography_labels)

df

# checking class imbalances for categorical factors
cat = df.columns[df.nunique()<10].tolist()
cat.pop()
print(cat)
plt.figure(figsize=(20,10))
i=1
for factor in cat:
    plt.subplot(2,3,i)
    sns.distplot(df[factor].loc[df['Exited']==0], hist=False)
    sns.distplot(df[factor].loc[df['Exited']==1], hist=False)
    i+=1

# train test split
x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=['Exited']).values, df.Exited.values,test_size=0.2, random_state=42)

def result_grid(x_train, y_train, x_test, y_test):
    model_param_grid={
        'Logistic Regression':(LogisticRegression(max_iter=10000),{'C': [0.01, 0.1, 1, 10.0],
                                                                   'solver': ['liblinear','lbfgs','newton-cg']}),
        'KNN':(KNeighborsClassifier(),{'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]}),
        'Decision Tree':(DecisionTreeClassifier(max_depth=10),{'criterion': ['gini','entropy'], 'splitter':['best','random'],
                                                  'max_depth':[10,50,100,150,200,250,300,350,400]}),
        'Random Forest':(RandomForestClassifier(),{'n_estimators':[50,100,150,200]})
    }

    results=[]

    for name, (model,parameters) in model_param_grid.items():
        grid=GridSearchCV(model,parameters)

        print(f"Tuning for {name}")

        st=time.time()

        grid.fit(x_train,y_train) # fitting in GridSearchCV
        y_pred=grid.predict(x_test) # predicts using best hyperparameters
        acc=accuracy_score(y_test,y_pred) # best accuracy obtained

        en=time.time()

        net_time=en-st

        # printing best parameters:
        print(f"Best hyperparameters for {name}: {grid.best_params_}")
        print(f"Best obtined score for {name}: {grid.best_score_*100:.3f}%")
        print(f"Running time for {name}:{net_time:.3f}s")

        results.append([name,grid.best_params_,grid.best_score_*100,net_time])
        print('-'*100)
        print()

    result_df=pd.DataFrame(results, columns=['model_name','best_parameters','best_test_score','running_time'])
    result_df=result_df.style.highlight_max(subset=['best_test_score'], color = 'lightgreen')
    result_df=result_df.highlight_min(subset=['best_test_score'], color = 'pink')
    result_df=result_df.highlight_max(subset=['running_time'], color = 'pink')
    result_df=result_df.highlight_min(subset=['running_time'], color = 'lightgreen')
    return result_df

res = result_grid(x_train, y_train, x_test, y_test)
res