
# Commented out IPython magic to ensure Python compatibility.
# Import necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import datetime as dt
# Remove deprecation warnings
import warnings
warnings.filterwarnings('ignore')
import pickle

chennai = pd.read_csv("https://raw.githubusercontent.com/githubshathik/MY_Machine_Learning_Project_Chennai_House_price_prediction/shathik/train-chennai-sale.csv")
chennai=pd.DataFrame(chennai)
chennai
chennai.describe().T
chennai.info()
chennai.isnull().sum().T
chennai=chennai.drop(columns='PRT_ID')

## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in chennai.columns if chennai[features].isnull().sum()>0]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(chennai[feature].isnull().mean(), 4),  ' % missing values')

chennai=chennai.dropna(how='any',axis=0)
chennai.info()

chennai=chennai.drop_duplicates()
chennai.info()

chennai1=chennai.copy()

chennai["DATE_SALE"] = pd.to_datetime(chennai["DATE_SALE"])
chennai['DATE_BUILD']=pd.to_datetime(chennai["DATE_BUILD"])

chennai['Yearsold']=chennai['DATE_SALE'].dt.year
chennai['Yearbuild']=chennai['DATE_BUILD'].dt.year

chennai['Houselife']=chennai['Yearsold']-chennai['Yearbuild']

chennai=chennai.drop(columns=["DATE_SALE","DATE_BUILD"])
# list of categorical variables
categorical_features = [feature for feature in chennai.columns if chennai[feature].dtypes == 'O']
categorical_features=chennai[categorical_features]
for feature in categorical_features:
  print('The feature is {} and number of categories are  = {}'.format(feature,(chennai[feature].unique())))

chennai['PARK_FACIL'] = chennai['PARK_FACIL'].replace('Noo','No')


chennai['STREET'] = chennai['STREET'].replace(['Pavd','NoAccess'],['Paved','No Access'])


chennai['BUILDTYPE'] = chennai['BUILDTYPE'].replace(['Other','Comercial'],['Others','Commercial'])

chennai['SALE_COND'] = chennai['SALE_COND'].replace(['Ab Normal','Partiall','Adj Land','PartiaLl'],['AbNormal','Partial','AdjLand','Partial'])


chennai['UTILITY_AVAIL'] = chennai['UTILITY_AVAIL'].replace(['All Pub','NoSewr ','NoSewr'],['AllPub','NoSeWa','NoSeWa'])

chennai['AREA'] = chennai['AREA'].replace(['TNagar','Chrompt','Chrmpet','Karapakam','Ana Nagar','Chormpet','Adyr','Velchery','Ann Nagar','KKNagar'],
                                          ['T Nagar','Chrompet','Chrompet','Karapakkam','Anna Nagar','Chrompet','Adyar','Velachery','Anna Nagar','KK Nagar'])

mean = chennai['SALES_PRICE'].mean()
std = chennai['SALES_PRICE'].std()
skew = chennai['SALES_PRICE'].skew()
print('SalePrice : mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}'.format(mean, std, skew))
chennai['SALES_PRICE1'] = np.log(chennai['SALES_PRICE'])

chennai["REG_FEE"]=np.log(chennai["REG_FEE"])

chennai["COMMIS"]=np.log(chennai["COMMIS"])

mean =chennai['SALES_PRICE1'].mean()
std = chennai['SALES_PRICE1'].std()
skew = chennai['SALES_PRICE1'].skew()
print('SalePrice : mean: {0:.4f}, std: {1:.4f}, skew: {2:.4f}'.format(mean, std, skew))

chennai=chennai.drop(columns=['SALES_PRICE'],axis=0)

co=chennai.corr().round(1)
# annot = True to print the values inside the square
# list of numerical variables
numerical_features = [feature for feature in chennai.columns if chennai[feature].dtypes != 'O']

# visualise the numerical variables
numerical_features=chennai[numerical_features]
numerical_features
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(chennai[feature].unique())<25 ]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))


chennai=chennai.drop(columns=["REG_FEE","COMMIS",'QS_ROOMS','QS_BEDROOM','QS_OVERALL','N_BEDROOM','QS_BATHROOM',"Yearsold","N_ROOM","DIST_MAINROAD"])

chennai.head()

cat=chennai[['AREA','UTILITY_AVAIL','PARK_FACIL','BUILDTYPE','STREET','SALE_COND','MZZONE']]

num=chennai[["INT_SQFT","SALES_PRICE1","Yearbuild","Houselife"]]
import pandas
from sklearn import preprocessing
cat['PARK_FACIL'] = cat['PARK_FACIL'].replace(["Yes","No"],[1,0])
cat['AREA'] = cat['AREA'].replace(["Karapakkam","Adyar","Chrompet","Velachery","KK Nagar","Anna Nagar","T Nagar"],[0,1,2,3,4,5,6])
cat['BUILDTYPE'] = cat['BUILDTYPE'].replace(["House","Others","Commercial"],[0,1,2])
cat['UTILITY_AVAIL'] = cat['UTILITY_AVAIL'].replace(["ELO","NoSeWa","AllPub"],[0,1,2])
cat['STREET'] = cat['STREET'].replace(["No Access","Paved","Gravel"],[0,1,2])
cat['SALE_COND'] = cat['SALE_COND'].replace(["Partial","Family","AbNormal","Normal Sale","AdjLand"],[0,1,2,3,4])
cat['MZZONE'] = cat['MZZONE'].replace(["A","C","I","RH","RL","RM"],[0,1,2,3,4,5])

X=pd.merge(cat,num,left_index = True, right_index = True)
X

X.shape

Y=X['SALES_PRICE1']
Y

X=X.drop(columns='SALES_PRICE1')
X

# Input Data
# X = features

# Output Data
# Y = SALESPRICE


# splitting data to training and testing dataset.

#from sklearn.cross_validation import train_test_split
#the submodule cross_validation is renamed and reprecated to model_selection
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2,random_state = 1)

print("xtrain shape : ", X_train.shape)
print("xtest shape : ", X_test.shape)
print("ytrain shape : ", Y_train.shape)
print("ytest shape : ", Y_test.shape)
XP=X_test.copy()


import xgboost as xgb
from sklearn.model_selection import cross_val_score

model1 = xgb.XGBRegressor(learning_rate = 0.3, n_estimators=100) # initialise the model
model1.fit(X_train,Y_train) #train the model

pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

# df1 = pd.DataFrame({'Actual_price': Y_test, 'Predicted_price': Y_pred})
# df1

# XP.head()

# df1["yearsale"]=XP['Yearbuild']+XP["Houselife"]
# df1

# from datetime import date
# df1["yearpassed"]=(date.today().year)-df1["yearsale"]
# df1

# df1["Predicted_price"]=np.exp(df1["Predicted_price"].values)
# df1["Actual_price"]=np.exp(df1["Actual_price"].values)
# df1

# pd.set_option('display.float_format', '{:.2f}'.format)

# #what is inflation?
# url :https://en.wikipedia.org/wiki/Inflation_in_India

# Because of the rate of inflation here we need to adjust the sales price for today rate
# Notably, the RBI had projected **CPI inflation at 5.3 percent** for fiscal year **2021-22**. This includes a projection of **5.1** percent in the second quarter,** 4.5 **percent in third,** 5.8** percent in the last quarter of the fiscal, with risks broadly balanced.

# assume **5%**rate of inflation and get the value of today price,

# # **Price today=Price historical*(1+ rate of inflation)^(years passed)**

# df1["Today_price"]=(df1["Predicted_price"])*((1+0.05)**(df1["yearpassed"]))
# df1