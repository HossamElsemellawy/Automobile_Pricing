import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(data):
## Returns a list of dicts from .csv file passed in as data"""
    with open(data, "rb") as f:
        reader = list(unicodecsv.DictReader(f))
    return reader

folderPath = "D:\\Data Science\\Auto1\\Model2\\"
file = read_csv(folderPath+'Auto1-DS-TestData.csv') # call the function and pass in your 'filename'

df_Auto = pd.DataFrame(file) 

df_Auto = df_Auto.replace('?',np.nan)

#####Convert Numeric column from Object to Numeric
df_Auto[['symboling','normalized-losses','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']] = \
df_Auto[['symboling','normalized-losses','wheel-base','length','width','height','curb-weight','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']].apply(pd.to_numeric)

#####Combine Categorical Variables###
##Num of Cylinders (4- and 4+)
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('twelve','5+')
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('eight','5+')
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('six','5+')
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('five','5+')

df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('four','4-')
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('two','4-')
df_Auto['num-of-cylinders'] = df_Auto['num-of-cylinders'].replace('three','4-')

#####Combine non frequent auto makers per cars price level
###Low Price Cars <= 10K
df_Auto['make'] = df_Auto['make'].replace('chevrolet','Low')
df_Auto['make'] = df_Auto['make'].replace('plymouth','Low')
df_Auto['make'] = df_Auto['make'].replace('isuzu','Low')
df_Auto['make'] = df_Auto['make'].replace('renault','Low')
####Mid Price Cars 10K ~ 20K
df_Auto['make'] = df_Auto['make'].replace('saab','Mid')
df_Auto['make'] = df_Auto['make'].replace('alfa-romero','Mid')
df_Auto['make'] = df_Auto['make'].replace('mercury','Mid')
df_Auto['make'] = df_Auto['make'].replace('audi','Mid')
####High Price Cars 20k+
df_Auto['make'] = df_Auto['make'].replace('porsche','High')
df_Auto['make'] = df_Auto['make'].replace('jaguar','High')

df_Auto['num-of-doors'] = df_Auto['num-of-doors'].replace(np.nan,'four')

###Combine body almost similar body style
df_Auto['body-style'] = df_Auto['body-style'].replace('convertible','hardtop')

df_Auto['fuel-system'] = df_Auto['fuel-system'].replace('spdi','OTH')
df_Auto['fuel-system'] = df_Auto['fuel-system'].replace('4bbl','OTH')
df_Auto['fuel-system'] = df_Auto['fuel-system'].replace('spfi','OTH')
df_Auto['fuel-system'] = df_Auto['fuel-system'].replace('mfi','OTH')

df_Auto = df_Auto.drop(['engine-type','stroke'], axis=1)

######Add ID Column###############
df_Auto.insert(0, 'Car_ID', range(1, 1 + len(df_Auto)))

####Add Interaction Variables###########
df_Auto['Aspire-FuelType'] = df_Auto['aspiration'] +'-'+ df_Auto['fuel-type']
df_Auto['Aspire-Doors'] = df_Auto['aspiration'] +'-'+ df_Auto['num-of-doors']

###df_Auto.to_csv('D:\Data Science\Auto1\out.csv')

##### Remove Rows with Null Price########
df_NotNull = df_Auto[df_Auto['price'].notnull()]

########Split Data into Dependent Dataframe(Y) and Independent Dataframe(X) Columns
df_Y = df_NotNull[['price']]
df_X_Tmp = df_NotNull[['aspiration','body-style','bore','city-mpg','compression-ratio','curb-weight','drive-wheels','engine-location','engine-size','fuel-system','fuel-type','height','highway-mpg','horsepower','length','make','normalized-losses','num-of-cylinders','num-of-doors','peak-rpm','symboling','wheel-base','width','Aspire-FuelType','Aspire-Doors']]

#######Build Dummy Varaibles for Categorical Columns#########
df_X = pd.get_dummies(df_X_Tmp)
df_X['make'] = df_X_Tmp['make']
######Transform Curb-Weight column and peak RPM column#########
df_X['curb-weight'] = df_X['curb-weight']/100
df_X['peak-rpm'] = df_X['peak-rpm']/100

########Split Data into Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state =4444)

####Prepare Dataframe for Null Replacement########
df_Null_Rep = X_train.groupby(['make']).agg({'bore': 'mean', 'normalized-losses': 'mean', 'horsepower': 'mean', 'peak-rpm': 'mean'})
df_Null_Rep = df_Null_Rep.rename(index=str, columns={"bore": "bore_Mean", "normalized-losses": "normalized-losses_Mean", "horsepower": "horsepower_Mean","peak-rpm": "peak-rpm_Mean"})

X_train = X_train.join(df_Null_Rep, on='make')
X_train.bore.fillna(X_train.bore_Mean, inplace=True)
X_train.horsepower.fillna(X_train.horsepower_Mean, inplace=True)
X_train['normalized-losses'].fillna(X_train['normalized-losses_Mean'], inplace=True)
X_train['peak-rpm'].fillna(X_train['peak-rpm_Mean'], inplace=True)

#####Remove Categorical Columns###########
X_train = X_train.drop(['make', 'bore_Mean','normalized-losses_Mean','horsepower_Mean','peak-rpm_Mean'], axis=1)

X_train = X_train.fillna(X_train.mean())

########Train Linear Regression Model#########
execfile(folderPath+"VariableSel.py")
selectedVar ,model = stepwise_selection(X_train, y_train, threshold_in=0.2, threshold_out = 0.3)

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train[selectedVar], y_train)

######Plot Traing Data########
plt.scatter(y_train, regr.predict(X_train[selectedVar]),  color='black')
plt.title('Train Data')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.xticks(())
plt.yticks(())
 
plt.show()

######Prepare Test Data#########
X_test = X_test.join(df_Null_Rep, on='make')
X_test.bore.fillna(X_test.bore_Mean, inplace=True)
X_test.horsepower.fillna(X_test.horsepower_Mean, inplace=True)
X_test['peak-rpm'].fillna(X_test['peak-rpm_Mean'], inplace=True)
X_test['normalized-losses'].fillna(X_test['normalized-losses_Mean'], inplace=True)

X_test = X_test.drop(['make', 'bore_Mean','normalized-losses_Mean','horsepower_Mean','peak-rpm_Mean'], axis=1)

X_test = X_test.fillna(X_train.mean())

######Plot Test Data########
predictions = regr.predict(X_test[selectedVar])
Train_Pred = regr.predict(X_train[selectedVar])

import seaborn as sns
sns.set()
plt.style.use('ggplot')

plt.scatter(y_test, predictions,  color='red')
plt.title('Test Data')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.xticks(())
plt.yticks(())
 
plt.show()

######Plot Residuals Data########
res = y_test - predictions
plt.scatter(y_test, res,  color='black')
plt.title('Test Data')
plt.xlabel('Actual Price')
plt.ylabel('Residuals')
plt.xticks(())
plt.yticks(())
plt.axhline(y=0.2, xmin=0, xmax=5000, linewidth=2, color='r')
##fig, ax = plt.subplots()
###ax.plot(x, y)
##ax.hlines(y=0.2, xmin=0, xmax=5000, linewidth=2, color='r')

plt.show()

bin_edge = [-3500,-2500,-1500,-500,500,1500,2500,3500,4500]
_ = plt.hist(res['price'],bins = bin_edge,density=True,)
_ = plt.xlabel("Actual Price - Predicted Price")
_ = plt.ylabel("Percentage of Cars")
plt.show()

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

print "Train Data RMSE: "+str(sqrt(mean_squared_error(y_train, Train_Pred)))
print "Train Data R-Square: "+repr(r2_score(y_train, Train_Pred))
print "==============================="
print "Test Data RMSE: "+str(sqrt(mean_squared_error(y_test, predictions)))
print "Test Data R-Square: "+repr(r2_score(y_test, predictions))
