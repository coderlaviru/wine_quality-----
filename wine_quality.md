```python
import numpy as np
import pandas as pd

#Importing tools for visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Import evaluation metric librarie s
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.preprocessing import LabelEncoder
#Libraries used for data  prprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

#Library used for ML Model implementation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
#import xgboost as xgb #Xtreme Gradient Boosting 
#librries used for ignore warnings 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
wine = pd.read_csv(r"C:\Users\lakshita\Desktop\datasets\winequality-red.csv")
print("Successfully Imported Data!")
```

    Successfully Imported Data!
    


```python
wine.head
```




    <bound method NDFrame.head of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0               7.4             0.700         0.00             1.9      0.076   
    1               7.8             0.880         0.00             2.6      0.098   
    2               7.8             0.760         0.04             2.3      0.092   
    3              11.2             0.280         0.56             1.9      0.075   
    4               7.4             0.700         0.00             1.9      0.076   
    ...             ...               ...          ...             ...        ...   
    1594            6.2             0.600         0.08             2.0      0.090   
    1595            5.9             0.550         0.10             2.2      0.062   
    1596            6.3             0.510         0.13             2.3      0.076   
    1597            5.9             0.645         0.12             2.0      0.075   
    1598            6.0             0.310         0.47             3.6      0.067   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                    11.0                  34.0  0.99780  3.51       0.56   
    1                    25.0                  67.0  0.99680  3.20       0.68   
    2                    15.0                  54.0  0.99700  3.26       0.65   
    3                    17.0                  60.0  0.99800  3.16       0.58   
    4                    11.0                  34.0  0.99780  3.51       0.56   
    ...                   ...                   ...      ...   ...        ...   
    1594                 32.0                  44.0  0.99490  3.45       0.58   
    1595                 39.0                  51.0  0.99512  3.52       0.76   
    1596                 29.0                  40.0  0.99574  3.42       0.75   
    1597                 32.0                  44.0  0.99547  3.57       0.71   
    1598                 18.0                  42.0  0.99549  3.39       0.66   
    
          alcohol  quality  
    0         9.4        5  
    1         9.8        5  
    2         9.8        5  
    3         9.8        6  
    4         9.4        5  
    ...       ...      ...  
    1594     10.5        5  
    1595     11.2        6  
    1596     11.0        6  
    1597     10.2        5  
    1598     11.0        6  
    
    [1599 rows x 12 columns]>




```python
wine.tail
```




    <bound method NDFrame.tail of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0               7.4             0.700         0.00             1.9      0.076   
    1               7.8             0.880         0.00             2.6      0.098   
    2               7.8             0.760         0.04             2.3      0.092   
    3              11.2             0.280         0.56             1.9      0.075   
    4               7.4             0.700         0.00             1.9      0.076   
    ...             ...               ...          ...             ...        ...   
    1594            6.2             0.600         0.08             2.0      0.090   
    1595            5.9             0.550         0.10             2.2      0.062   
    1596            6.3             0.510         0.13             2.3      0.076   
    1597            5.9             0.645         0.12             2.0      0.075   
    1598            6.0             0.310         0.47             3.6      0.067   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                    11.0                  34.0  0.99780  3.51       0.56   
    1                    25.0                  67.0  0.99680  3.20       0.68   
    2                    15.0                  54.0  0.99700  3.26       0.65   
    3                    17.0                  60.0  0.99800  3.16       0.58   
    4                    11.0                  34.0  0.99780  3.51       0.56   
    ...                   ...                   ...      ...   ...        ...   
    1594                 32.0                  44.0  0.99490  3.45       0.58   
    1595                 39.0                  51.0  0.99512  3.52       0.76   
    1596                 29.0                  40.0  0.99574  3.42       0.75   
    1597                 32.0                  44.0  0.99547  3.57       0.71   
    1598                 18.0                  42.0  0.99549  3.39       0.66   
    
          alcohol  quality  
    0         9.4        5  
    1         9.8        5  
    2         9.8        5  
    3         9.8        6  
    4         9.4        5  
    ...       ...      ...  
    1594     10.5        5  
    1595     11.2        6  
    1596     11.0        6  
    1597     10.2        5  
    1598     11.0        6  
    
    [1599 rows x 12 columns]>




```python
wine.corr
```




    <bound method DataFrame.corr of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0               7.4             0.700         0.00             1.9      0.076   
    1               7.8             0.880         0.00             2.6      0.098   
    2               7.8             0.760         0.04             2.3      0.092   
    3              11.2             0.280         0.56             1.9      0.075   
    4               7.4             0.700         0.00             1.9      0.076   
    ...             ...               ...          ...             ...        ...   
    1594            6.2             0.600         0.08             2.0      0.090   
    1595            5.9             0.550         0.10             2.2      0.062   
    1596            6.3             0.510         0.13             2.3      0.076   
    1597            5.9             0.645         0.12             2.0      0.075   
    1598            6.0             0.310         0.47             3.6      0.067   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                    11.0                  34.0  0.99780  3.51       0.56   
    1                    25.0                  67.0  0.99680  3.20       0.68   
    2                    15.0                  54.0  0.99700  3.26       0.65   
    3                    17.0                  60.0  0.99800  3.16       0.58   
    4                    11.0                  34.0  0.99780  3.51       0.56   
    ...                   ...                   ...      ...   ...        ...   
    1594                 32.0                  44.0  0.99490  3.45       0.58   
    1595                 39.0                  51.0  0.99512  3.52       0.76   
    1596                 29.0                  40.0  0.99574  3.42       0.75   
    1597                 32.0                  44.0  0.99547  3.57       0.71   
    1598                 18.0                  42.0  0.99549  3.39       0.66   
    
          alcohol  quality  
    0         9.4        5  
    1         9.8        5  
    2         9.8        5  
    3         9.8        6  
    4         9.4        5  
    ...       ...      ...  
    1594     10.5        5  
    1595     11.2        6  
    1596     11.0        6  
    1597     10.2        5  
    1598     11.0        6  
    
    [1599 rows x 12 columns]>




```python
wine.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(wine.shape)
```

    (1599, 12)
    


```python
print(wine.isna().sum())
```

    fixed acidity           0
    volatile acidity        0
    citric acid             0
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      0
    sulphates               0
    alcohol                 0
    quality                 0
    dtype: int64
    


```python
wine.info
```




    <bound method DataFrame.info of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0               7.4             0.700         0.00             1.9      0.076   
    1               7.8             0.880         0.00             2.6      0.098   
    2               7.8             0.760         0.04             2.3      0.092   
    3              11.2             0.280         0.56             1.9      0.075   
    4               7.4             0.700         0.00             1.9      0.076   
    ...             ...               ...          ...             ...        ...   
    1594            6.2             0.600         0.08             2.0      0.090   
    1595            5.9             0.550         0.10             2.2      0.062   
    1596            6.3             0.510         0.13             2.3      0.076   
    1597            5.9             0.645         0.12             2.0      0.075   
    1598            6.0             0.310         0.47             3.6      0.067   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                    11.0                  34.0  0.99780  3.51       0.56   
    1                    25.0                  67.0  0.99680  3.20       0.68   
    2                    15.0                  54.0  0.99700  3.26       0.65   
    3                    17.0                  60.0  0.99800  3.16       0.58   
    4                    11.0                  34.0  0.99780  3.51       0.56   
    ...                   ...                   ...      ...   ...        ...   
    1594                 32.0                  44.0  0.99490  3.45       0.58   
    1595                 39.0                  51.0  0.99512  3.52       0.76   
    1596                 29.0                  40.0  0.99574  3.42       0.75   
    1597                 32.0                  44.0  0.99547  3.57       0.71   
    1598                 18.0                  42.0  0.99549  3.39       0.66   
    
          alcohol  quality  
    0         9.4        5  
    1         9.8        5  
    2         9.8        5  
    3         9.8        6  
    4         9.4        5  
    ...       ...      ...  
    1594     10.5        5  
    1595     11.2        6  
    1596     11.0        6  
    1597     10.2        5  
    1598     11.0        6  
    
    [1599 rows x 12 columns]>




```python
plt.figure(figsize=(8, 6))
sns.countplot(x=wine['quality'], palette='coolwarm')  # Use a vibrant palette
plt.title('Count of Wines by Quality', fontsize=16)
plt.xlabel('Quality', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for better readability
plt.show()
```


    
![png](output_9_0.png)
    



```python

sns.countplot(x=wine["alcohol"])
plt.show()
```


    
![png](output_10_0.png)
    



```python
sns.countplot(x=wine["volatile acidity"])
plt.show()
```


    
![png](output_11_0.png)
    



```python
sns.countplot(x=wine["fixed acidity"])
plt.show()
```


    
![png](output_12_0.png)
    



```python
sns.countplot(x=wine["density"])
plt.show()
```


    
![png](output_13_0.png)
    



```python
sns.countplot(x=wine["citric acid"])
plt.show()
```


    
![png](output_14_0.png)
    



```python
sns.countplot(x=wine["pH"])
plt.show()
```


    
![png](output_15_0.png)
    



```python
sns.countplot(x=wine["residual sugar"])
plt.show()
```


    
![png](output_16_0.png)
    



```python
sns.countplot(x=wine["chlorides"])
plt.show()
```


    
![png](output_17_0.png)
    



```python
sns.countplot(x=wine["free sulfur dioxide"])
plt.show()
```


    
![png](output_18_0.png)
    



```python
sns.kdeplot(wine.query('quality>2').quality)
```




    <Axes: xlabel='quality', ylabel='Density'>




    
![png](output_19_1.png)
    



```python
sns.distplot(x=wine["alcohol"])
```




    <Axes: ylabel='Density'>




    
![png](output_20_1.png)
    



```python
wine.plot(kind="box",subplots=True,layout=(4,4),sharex=False)
```




    fixed acidity              Axes(0.125,0.712609;0.168478x0.167391)
    volatile acidity        Axes(0.327174,0.712609;0.168478x0.167391)
    citric acid             Axes(0.529348,0.712609;0.168478x0.167391)
    residual sugar          Axes(0.731522,0.712609;0.168478x0.167391)
    chlorides                  Axes(0.125,0.511739;0.168478x0.167391)
    free sulfur dioxide     Axes(0.327174,0.511739;0.168478x0.167391)
    total sulfur dioxide    Axes(0.529348,0.511739;0.168478x0.167391)
    density                 Axes(0.731522,0.511739;0.168478x0.167391)
    pH                          Axes(0.125,0.31087;0.168478x0.167391)
    sulphates                Axes(0.327174,0.31087;0.168478x0.167391)
    alcohol                  Axes(0.529348,0.31087;0.168478x0.167391)
    quality                  Axes(0.731522,0.31087;0.168478x0.167391)
    dtype: object




    
![png](output_21_1.png)
    



```python
wine.plot(kind="density",subplots=True,layout=(4,4),sharex=False)
```




    array([[<Axes: ylabel='Density'>, <Axes: ylabel='Density'>,
            <Axes: ylabel='Density'>, <Axes: ylabel='Density'>],
           [<Axes: ylabel='Density'>, <Axes: ylabel='Density'>,
            <Axes: ylabel='Density'>, <Axes: ylabel='Density'>],
           [<Axes: ylabel='Density'>, <Axes: ylabel='Density'>,
            <Axes: ylabel='Density'>, <Axes: ylabel='Density'>],
           [<Axes: ylabel='Density'>, <Axes: ylabel='Density'>,
            <Axes: ylabel='Density'>, <Axes: ylabel='Density'>]], dtype=object)




    
![png](output_22_1.png)
    



```python
wine.hist(figsize=(10,10),bins=50)
plt.show()
```


    
![png](output_23_0.png)
    



```python
corr=wine.corr()
sns.heatmap(corr,annot=True)
```




    <Axes: >




    
![png](output_24_1.png)
    



```python
sns.pairplot(wine)
```




    <seaborn.axisgrid.PairGrid at 0x18d0ec823f0>




    
![png](output_25_1.png)
    



```python
sns.violinplot(x='quality',y='alcohol',data=wine)
```




    <Axes: xlabel='quality', ylabel='alcohol'>




    
![png](output_26_1.png)
    


# Feature selection


```python
#create classification version of target variable
wine['goodquality']=[1 if x >= 7 else 0 for x in wine ['quality']] # sepatated features varibles and 
#target
X=wine.drop(['quality','goodquality'],axis=1)
Y = wine ['goodquality']

```


```python
#seee poperties of good vs bad qualioty
wine['goodquality'].value_counts()
```




    goodquality
    0    1382
    1     217
    Name: count, dtype: int64




```python
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.880</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.760</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.280</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>6.2</td>
      <td>0.600</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.090</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99490</td>
      <td>3.45</td>
      <td>0.58</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>5.9</td>
      <td>0.550</td>
      <td>0.10</td>
      <td>2.2</td>
      <td>0.062</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>0.99512</td>
      <td>3.52</td>
      <td>0.76</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>1597</th>
      <td>5.9</td>
      <td>0.645</td>
      <td>0.12</td>
      <td>2.0</td>
      <td>0.075</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99547</td>
      <td>3.57</td>
      <td>0.71</td>
      <td>10.2</td>
    </tr>
    <tr>
      <th>1598</th>
      <td>6.0</td>
      <td>0.310</td>
      <td>0.47</td>
      <td>3.6</td>
      <td>0.067</td>
      <td>18.0</td>
      <td>42.0</td>
      <td>0.99549</td>
      <td>3.39</td>
      <td>0.66</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
<p>1599 rows × 11 columns</p>
</div>




```python
Y
```




    0       0
    1       0
    2       0
    3       0
    4       0
           ..
    1594    0
    1595    0
    1596    0
    1597    0
    1598    0
    Name: goodquality, Length: 1599, dtype: int64



# Feature Importance


```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=7)
```


```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print("Acuuracy Score:",accuracy_score(Y_test,Y_pred))
```

    Acuuracy Score: 0.8708333333333333
    


```python
model=LogisticRegression()
classifiern=ExtraTreesClassifier()
classifiern.fit(X,Y)
score=classifiern.feature_importances_
print(score)
```

    [0.07730576 0.099888   0.0936217  0.07419059 0.06757834 0.06858845
     0.08246797 0.08588263 0.0652764  0.10972933 0.17547084]
    


```python
confusion_mat=confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)
```

    <function confusion_matrix at 0x0000018D088DA2A0>
    

# Using KNN


```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
```

    Accuracy Score: 0.8729166666666667
    

# SVC


```python
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))
```

    Accuracy Score: 0.86875
    

# Decision tree


```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
```

    Accuracy Score: 0.8645833333333334
    

# Using Guassian NB


```python
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))
```

    Accuracy Score: 0.8333333333333334
    

# Using Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))
```

    Accuracy Score: 0.89375
    

# Using xgBoost


```python
import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))
results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN', 'SVC','Decision Tree' ,'GaussianNB','Random Forest','Xgboost'],
    'Score': [0.870,0.872,0.868,0.864,0.833,0.893,0.879]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
```

    Accuracy Score: 0.8895833333333333
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
    </tr>
    <tr>
      <th>Score</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.893</th>
      <td>Random Forest</td>
    </tr>
    <tr>
      <th>0.879</th>
      <td>Xgboost</td>
    </tr>
    <tr>
      <th>0.872</th>
      <td>KNN</td>
    </tr>
    <tr>
      <th>0.870</th>
      <td>Logistic Regression</td>
    </tr>
    <tr>
      <th>0.868</th>
      <td>SVC</td>
    </tr>
    <tr>
      <th>0.864</th>
      <td>Decision Tree</td>
    </tr>
    <tr>
      <th>0.833</th>
      <td>GaussianNB</td>
    </tr>
  </tbody>
</table>
</div>



## Hence I will use Random Forest algorithms for training my model.
