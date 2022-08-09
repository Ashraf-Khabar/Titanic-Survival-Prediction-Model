```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

# <span style='color:#3C9DD0'>Regression with SKLEARN</span>


```python
np.random.seed(0)
m = 100 # creating 100 sample
X = np.linspace(0,10,m).reshape(m,1)
y = X + np.random.randn(m,1)
```


```python
plt.scatter(X,y,c='y')
```




    <matplotlib.collections.PathCollection at 0x16c247d0a90>




    
![png](output_3_1.png)
    



```python
from sklearn.linear_model import LinearRegression
```


```python
model = LinearRegression()
model.fit(X,y)
model.score(X,y)

predictions = model.predict(X)

plt.scatter(X,y,c=y)
plt.plot(X,predictions, c = 'r')
```




    [<matplotlib.lines.Line2D at 0x16c2493c310>]




    
![png](output_5_1.png)
    


# <span style='color:#3C9DD0'>Classificaion problem with with SKLEARN</span>


```python
titanic = sns.load_dataset('titanic')
titanic.shape
titanic.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic = titanic[['survived','pclass','sex','age']]
titanic.dropna(axis=0 , inplace=True)
titanic['sex'].replace(['male','female'] , [0,1] , inplace=True)
titanic.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
model = KNeighborsClassifier()
```


```python
y = titanic['survived']
X = titanic.drop('survived' , axis=1)
```


```python
X.head()
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
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y.head()
```




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: survived, dtype: int64




```python
model.fit(X,y)
print(f'{model.score(X,y)*100} %')
```

    84.17366946778712 %
    

### <span style='color:#C90076'>The function which will say that if we will survive or not</span>


```python
def survived(model , pclass=3 , sex=0 , age=22):
    x = np.array([pclass,sex,age]).reshape(1,3)
    if model.predict(x) == [0]:
        print('You will not servive')
    else:
        print('You will survive')
```


```python
survived(model)
```

    You will not servive
    

### <span style='color:#C90076'>The probability of surviving (vice versa) </span>


```python
def survived(model , pclass=3 , sex=0 , age=22):
    x = np.array([pclass,sex,age]).reshape(1,3)
    if model.predict(x) == [0]:
        print('You will not servive')
        print(f'you will not survive with {model.predict_proba(x)[0,0]*100}% and survive with {model.predict_proba(x)[0,1]*100}%')
    else:
        print('You will survive')
        print(f'you will not survive with {model.predict_proba(x)[0,0]*100}% and survive with {model.predict_proba(x)[0,1]*100}%')
```


```python
survived(model,3,1,25)
```

    You will not servive
    you will not survive with 80.0% and survive with 20.0%
  
