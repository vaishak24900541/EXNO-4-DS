# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
### NAME:VAISHAK.M
### REG NO:212224040355

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
<img width="1751" height="723" alt="image" src="https://github.com/user-attachments/assets/6c1a2406-efd1-42bb-8e05-c67656868b57" />

```
data.isnull().sum()
```
<img width="667" height="612" alt="image" src="https://github.com/user-attachments/assets/ae3e16bc-21f1-41e7-b32b-4d7c605d1133" />

```
missing=data[data.isnull().any(axis=1)]
missing
```
<img width="1709" height="513" alt="image" src="https://github.com/user-attachments/assets/584f7a9b-0e64-49e8-9602-fca2ca0869b7" />

```
data2=data.dropna(axis=0)
data2
```
<img width="1740" height="742" alt="image" src="https://github.com/user-attachments/assets/028e4dfa-bb21-4509-8f12-7e931d4dea8b" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
<img width="1413" height="413" alt="image" src="https://github.com/user-attachments/assets/99bbe11d-8d26-4c2e-8866-79f2b730ab42" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="472" height="533" alt="image" src="https://github.com/user-attachments/assets/ed0bddc9-44fc-4d49-86e1-42d865256c3e" />

```
data2
```
<img width="1602" height="520" alt="image" src="https://github.com/user-attachments/assets/e28f186c-7b17-4ac1-ae0c-56367ae891ce" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="1759" height="613" alt="image" src="https://github.com/user-attachments/assets/d68c8791-206a-495a-aa5b-b3d8ce985500" />

```
columns_list=list(new_data.columns)
print(columns_list)
```
<img width="1759" height="55" alt="image" src="https://github.com/user-attachments/assets/76dd754b-010b-4c75-90be-cb24cff21660" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

```
y=new_data['SalStat'].values
print(y)
```
<img width="297" height="44" alt="image" src="https://github.com/user-attachments/assets/b69c29b0-46be-4957-a477-5bc644d2c0b0" />

```
x=new_data[features].values
print(x)
```
<img width="610" height="187" alt="image" src="https://github.com/user-attachments/assets/e537587f-b5b1-4b85-b590-d8cbf00a4b93" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="523" height="89" alt="image" src="https://github.com/user-attachments/assets/41d6a1c0-8595-4df0-91ba-031a42cb3fa6" />

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
<img width="323" height="80" alt="image" src="https://github.com/user-attachments/assets/d5f6fd4a-f375-4094-a89b-3ac2fdcf7ccd" />

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
<img width="371" height="53" alt="image" src="https://github.com/user-attachments/assets/1b266ba9-b081-4002-9c1f-e588b3207448" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="405" height="56" alt="image" src="https://github.com/user-attachments/assets/60bffda8-f900-4ae6-9435-17f76aca4d3a" />

```
data.shape
```
<img width="222" height="36" alt="image" src="https://github.com/user-attachments/assets/f6ee481c-bbae-42a6-8433-2b56751d20bf" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1753" height="105" alt="Screenshot 2025-09-30 114402" src="https://github.com/user-attachments/assets/c881a189-a57e-4b04-aa60-e0e567a5e87f" />

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
<img width="651" height="262" alt="image" src="https://github.com/user-attachments/assets/f275e674-1c22-46a9-b6d1-bdbeabcfa41b" />

```
tips.time.unique()
```
<img width="557" height="69" alt="image" src="https://github.com/user-attachments/assets/829e743c-8a71-4dab-91e4-c7e59f1b3029" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="303" height="105" alt="image" src="https://github.com/user-attachments/assets/d9f32c12-7a60-428b-877f-ac49ad00030a" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
<img width="510" height="54" alt="image" src="https://github.com/user-attachments/assets/d6ee74d2-0bb0-43b0-bc10-3b22367b6c48" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
