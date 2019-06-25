# Predictive Maintenance of Air Quality Data

```python
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```


```python
# Load dataset
sensor_file = "./data/sensor_data.csv"
quality_file = "./data/quality_control_data.csv"
# names = ['weight', 'humidity', 'temperature', 'quality']
sensor_data = pandas.read_csv(sensor_file)
quality_data = pandas.read_csv(quality_file)
```


```python
sensor_data.head(10)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>humidity</th>
      <th>temperature</th>
      <th>prod_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1030.871118</td>
      <td>29.687881</td>
      <td>71.995808</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1044.961148</td>
      <td>28.862453</td>
      <td>68.468664</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>972.710479</td>
      <td>37.951588</td>
      <td>65.121344</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1010.182509</td>
      <td>25.076383</td>
      <td>67.821336</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>970.039236</td>
      <td>27.137886</td>
      <td>72.931800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>990.154359</td>
      <td>32.422428</td>
      <td>71.406207</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>965.660243</td>
      <td>42.603619</td>
      <td>65.876158</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>969.221212</td>
      <td>31.655071</td>
      <td>74.430054</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>976.495532</td>
      <td>26.499721</td>
      <td>69.866121</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>974.993517</td>
      <td>38.644055</td>
      <td>69.891709</td>
      <td>10</td>
    </tr>
  </tbody>
</table>



```python
quality_data.head(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prod_id</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>good</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>poor</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>good</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>good</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>good</td>
    </tr>
  </tbody>
</table>
</div>




```python
rawdataset = sensor_data.merge(quality_data, on="prod_id")
```


```python
rawdataset.head(5)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>humidity</th>
      <th>temperature</th>
      <th>prod_id</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1030.871118</td>
      <td>29.687881</td>
      <td>71.995808</td>
      <td>1</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1044.961148</td>
      <td>28.862453</td>
      <td>68.468664</td>
      <td>2</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>972.710479</td>
      <td>37.951588</td>
      <td>65.121344</td>
      <td>3</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1010.182509</td>
      <td>25.076383</td>
      <td>67.821336</td>
      <td>4</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>970.039236</td>
      <td>27.137886</td>
      <td>72.931800</td>
      <td>5</td>
      <td>good</td>
    </tr>
  </tbody>
</table>



```python
dataset = rawdataset.drop(columns='prod_id')
dataset.head(10)
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
      <th>humidity</th>
      <th>temperature</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1030.871118</td>
      <td>29.687881</td>
      <td>71.995808</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1044.961148</td>
      <td>28.862453</td>
      <td>68.468664</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>972.710479</td>
      <td>37.951588</td>
      <td>65.121344</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1010.182509</td>
      <td>25.076383</td>
      <td>67.821336</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>970.039236</td>
      <td>27.137886</td>
      <td>72.931800</td>
      <td>good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>990.154359</td>
      <td>32.422428</td>
      <td>71.406207</td>
      <td>good</td>
    </tr>
    <tr>
      <th>6</th>
      <td>965.660243</td>
      <td>42.603619</td>
      <td>65.876158</td>
      <td>poor</td>
    </tr>
    <tr>
      <th>7</th>
      <td>969.221212</td>
      <td>31.655071</td>
      <td>74.430054</td>
      <td>good</td>
    </tr>
    <tr>
      <th>8</th>
      <td>976.495532</td>
      <td>26.499721</td>
      <td>69.866121</td>
      <td>good</td>
    </tr>
    <tr>
      <th>9</th>
      <td>974.993517</td>
      <td>38.644055</td>
      <td>69.891709</td>
      <td>good</td>
    </tr>
  </tbody>
</table>

```python
# shape
print(dataset.shape)
```

    (3000, 4)



```python
# descriptions
print(dataset.describe())
```

                weight     humidity  temperature
    count  3000.000000  3000.000000  3000.000000
    mean    999.940363    34.863581    69.962969
    std      28.765904     5.755869     2.857898
    min     950.017007    25.008023    65.000514
    25%     975.552942    29.783650    67.522238
    50%     998.875197    34.825848    69.890808
    75%    1025.649219    39.887405    72.414522
    max    1049.954013    44.986735    74.999312



```python
# quality distribution
print(dataset.groupby('quality').size())
```

    quality
    good    2907
    poor      93
    dtype: int64



```python
# box and whisker plots to show data distribution
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
```


![png](output_10_0.png)



```python
# check the histograms
dataset.hist()
plt.show()
```


![png](output_11_0.png)



```python
# scatter plot matrix - anything useful here?
scatter_matrix(dataset)
plt.show()
```


![png](output_12_0.png)



```python
# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.20
seed = 8
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
```


```python
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
```


```python
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```

    LR: 0.976667 (0.008375)
    LDA: 0.973750 (0.007229)
    KNN: 0.992083 (0.005417)
    CART: 0.998750 (0.002668)
    NB: 0.994167 (0.003333)
    SVM: 0.985417 (0.005966)



```python
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Comparison of ML Models')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```


![png](output_16_0.png)



```python
# Make predictions on validation dataset
#knn = KNeighborsClassifier()
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
```

    0.9983333333333333
    [[581   0]
     [  1  18]]
                  precision    recall  f1-score   support
    
            good       1.00      1.00      1.00       581
            poor       1.00      0.95      0.97        19
    
        accuracy                           1.00       600
       macro avg       1.00      0.97      0.99       600
    weighted avg       1.00      1.00      1.00       600


​    

# Now test some values of your own


```python
testWeight = 1200
testHumidity = 60
testTemperature = 65
testPrediction = CART.predict([[testWeight,testHumidity,testTemperature]])
print(testPrediction)
```

    ['good']

