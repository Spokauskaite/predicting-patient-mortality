# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:33:16 2020

@author: Lina
"""

###################################################################################################
#
#                                PREDICTING PATIENT MORTALITY
#
###################################################################################################

# We analyzed EHR(Electronic Health Record) Dream Challenge data with the goal to predict which patient
# would die in six months after the last visit to the doctor's office. We could not share the data, as 
# it is against the rules of Dream Challenges, but we can share the model building steps.
# Initially we received 7 data tables with the number of observations ranging between
# 97,000-11 mln. Demographics table had 97,000 observations representing each patient. 
# We manually engineered features and created one table with 97,000 observations(one for each patient)
# and ~15,000 features. The reason dimentionality column-wise grew significantly, is because dummy 
# variables were created for big part the of features. These features were indicators of drugs patient used, 
# health conditions patient had, and procedures patient received. The other variables were 
# measurements(continuous), observations(categorical) and demographics(categorical and discreet numerical).
# We also created "died_in_six_months" indicator variable that we used as a response variable in the model. As 
# this was a binary outcome, we built a classification model. There were various classification algrithms
# we could have used, including tree models, but we expected that tree models would not
# perform well for this data, because of large number of dummy variables. Logistic regression had best 
# preformance, so we used it for our final model.

# In addition to previously described data, we also received test data of 23,000 patients in the same setting.
# We used it to evaluate our final model. 

# #################################################################################################


################# LOADING MODULES AND READING DATA ################################################

#load packages
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from category_encoders import LeaveOneOutEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegressionCV

# read data
train_data = pd.read_csv('train_data.csv',low_memory=False)
test_data  = pd.read_csv('test_data.csv',low_memory=False)

# We separated variable 'died_in_six_months' as this is our response variable, 
# an indicator variable indicating whether patient died in  6months after the last visit  to 
# the doctors office.

y = train_data['died_in_six_months']
X = train_data.drop(['died_in_six_months','person_id'],axis=1)

# We split our training data further into training and testing data. We used this
# testing data to compare various models we created. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=5)


########################## PRE-PROCESSING #################################################
# We  separated our variables into different groups based on their data types, so we could
# pre-process them accordingly.
categorical_features = X.loc[:,X.dtypes == 'object'].columns
numeric_features = X.loc[:,(X.dtypes == 'float64') | (X.columns =='age')].columns
indicator_features = X.loc[:,(X.dtypes == 'int64') & (X.columns != 'age')].columns

# For numeric features we first imputed missing data using median. We choose median over mean,
# because our data is not normalized yet. It could have been skewed and had outliers, which would
# have given biased mean estimate.
# After imputing missing data, we used quantile transformer. This transformation method is robust to
# outliers, and transforms variables so they have normal distribution and 
# all have similar range. 

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', QuantileTransformer(output_distribution='normal', random_state=0))])

# For categorical features we imputed missing data with the most frequent value of the column.
# After that we encoded these variables using bayesian encoder LeaveOneOutEncoder. We chose this encoder 
# because our categorical variables were of high cardinality.

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('leaveoneout', LeaveOneOutEncoder(return_df=False))
    ])

# for Indicator variables we imputed missing data with 0, as they only have values
# 0 and 1, 1 for event occuring)
indicator_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))])

# we used column transformer to transform all the data based on  variable type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('ind', indicator_transformer, indicator_features)])


######################## SAMPLING ######################################################### 

# Our data was severely imbalanced, we had ratio of 1:200 of the event happening. 
# Some models have options to specify weights or can have parameter set to class_weight='balanced'
# to balance the data. We chose to do oversampling for the minority group. 
# We used algorithm SMOTE (Synthetic Minority Oversampling TEchnique). It
# randomly picks a point from the minority class and computes the k-nearest neighbors for this point. 
# The synthetic points are added between the chosen point and its neighbors. After applying
# SMOTE, we had a balanced dataset.

sampler = SMOTE(ratio='minority',random_state=0)

################################# MODEL ##################################################

# Finally, we built logistic regression model. We used LogisticRegressionCV, because it 
# had a built-in cross-validation. 
classifier = LogisticRegressionCV(cv=5, scoring='roc_auc', random_state=5,max_iter=1000)
                                  


#######################  CREATING PIPELINE ###############################################

# We added all the preprocessing steps and classifier into the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor)
                    ,('sampler', sampler)
                    ,('classifier', classifier)
                      ])
    

#########################  FITTING THE MODEL AND PREDICTING ##############################
# we fit the model
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

########################  METRICS  ######################################################
# we see how well our model is at making predictions

#              confusion matrix

# we can see how our model classified the items.
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

#                 ROC CURVE

# we draw a roc curve and look at auc score. Straight line indicates that model is not 
# classifying well. The closer to 1 the auc score is, the better the model. 
import matplotlib.pyplot  as plt
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# We can also look at the report. We can see f1 score for the event haappening. We use it to 
# compare the models
print(metrics.classification_report(y_test, y_pred))
