## Neelanjan - Lead sourceing case study_1


# importing libs :

import pandas as pd, numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import *

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve

## 1. Data reading & understanding :

df_leadscore = pd.read_csv(r'D:\Data Science\Machine Learning\Education Company Case study\Lead Scoring Assignment\Leads.csv')
df_leadscore.head()

## Data inspection :


df_leadscore.shape

df_leadscore.describe()

df_leadscore.info()

## 2. Data Peparation 

## 2a. Cleaning & mnipulation :

# Checking null values (%)

df_leadscore.isnull().mean()*100

# Lets get the list of teh col having missig values over 45% 
c1 = df_leadscore.isnull().mean()*100 > 45
df_leadscore.columns[c1==True]

# Lets drop the cols having >45% missing values as those will not significantly contribute to our analisis
missing_45p = ['Lead Quality', 'Asymmetrique Activity Index',
       'Asymmetrique Profile Index', 'Asymmetrique Activity Score',
       'Asymmetrique Profile Score']
df_leadscore1 = df_leadscore.drop(labels=missing_45p,axis=1)
df_leadscore1.columns


# Getting a new data frame with missing values < 45% 
df_leadscore1.head()

df_leadscore1.shape

df_leadscore1.isnull().mean()*100

## Confirm that all in cols of our deta frame contain <45% of null values - so, we are Good to proceed further 

# Getting the columns which are not very much important for our analysis
print(df_leadscore1["Lead Number"].value_counts)
print('\n *************************************************** \n')
print(df_leadscore1["Prospect ID"].value_counts)
print('\n *************************************************** \n')

* Looks like Lead number and Prospect ID have all unique values which are not very much useful for our data analysis,so we can drop them.

df_leadscore2 = df_leadscore1.drop(["Lead Number" , "Prospect ID"] , axis= 1)
df_leadscore2.head()

df_leadscore2.shape

# Lets check the null value again :
df_leadscore2.isnull().mean()*100

** cool - we got the expected outcomes 

## Handeling "Select" Label in catagorical cols :

# converting select labeles to "NaN"

df_leadscore2 = df_leadscore2.replace('Select', np.nan)

# re-Checking the null values %
df_leadscore2.isnull().mean().sort_values(ascending=False)*100


We see that for some columns we have High Percentage of Missing Values. We can drop the columns with missing values greater than 45% .

# Lets get the list of teh col having missig values over 45% 
c2 = df_leadscore2.isnull().mean()*100 > 45
df_leadscore2.columns[c2==True]

# Lets drop the cols having >45% missing values as those will not significantly contribute to our analisis
missing_45p = ['How did you hear about X Education', 'Lead Profile']
df_leadscore3 = df_leadscore2.drop(labels=missing_45p,axis=1)
df_leadscore3.columns

df_leadscore3.head()

df_leadscore3.isnull().mean().sort_values(ascending= False)*100

# Separatimg catagorical & numerical cols : Logic = if more than 30 unique items are present in a column, 
# will be conceder as a count col , else it is catagorical 

cont_cols=[]
cat_cols=[]
for i in df_leadscore3.columns:
    if df_leadscore3[i].nunique()>30:
        print(i,df_leadscore3[i].nunique(),  "----cont_cols")
        cont_cols.append(i)
    else:
        print(i,df_leadscore3[i].nunique(),  "----cat_cols")
        cat_cols.append(i)


df_leadscore3.columns

## CHENGING 'country' from cont-col to cat_col :by observing the col values 

cont_cols=[ 'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
cat_cols= ['Lead Origin', 'Lead Source', 'Do Not Email', 'Do Not Call',
       'Converted', 'Last Activity', 'Specialization',
       'What is your current occupation','Country',
       'What matters most to you in choosing a course', 'Search', 'Magazine',
       'Newspaper Article', 'X Education Forums', 'Newspaper',
       'Digital Advertisement', 'Through Recommendations',
       'Receive More Updates About Our Courses', 'Tags',
       'Update me on Supply Chain Content', 'Get updates on DM Content',
       'City', 'I agree to pay the amount through cheque',
       'A free copy of Mastering The Interview', 'Last Notable Activity']

for i in cat_cols:
    print(i)
    print()
    print(df_leadscore3[i].value_counts())
    print()
    print("percentage",100*df_leadscore3[i].value_counts(normalize=True))
    print()


# Replacing the null values in categorical columns by mode of that respective the colmun
for i in cat_cols:
    if df_leadscore3[i].isnull().sum()>0:
        value=df_leadscore3[i].mode()[0]
        df_leadscore3[i]=df_leadscore3[i].fillna(value)


# Replacing the null values in continuous columns by median of that respective the colmun
for i in cont_cols:
    if df_leadscore3[i].isnull().sum()>0:   
        value=df_leadscore3[i].median()
        df_leadscore3[i]=df_leadscore3[i].fillna(value)


#now check all the null values are replaced or not
df_leadscore3.isnull().sum()

** Awasome - all null values are eleminated !
However, we still find, in some of the coloums, only 1 major values find, we can also drop them as they are not contributing much affect in analisis 
Those cols are Tags,Not Call, Search, Magazine, Newspaper Article, X Education Forums, Newspaper, Digital Advertisement, Through Recommendations, Receive More Updates About Our Courses, Update me on Supply Chain Content, Get updates on DM Content, I agree to pay the amount through cheque.

# Droping these cols :
df_leadscore4 = df_leadscore3.drop(['Country','What matters most to you in choosing a course','Do Not Call', 'Search', 'Magazine','Newspaper Article', 'X Education Forums','Newspaper',
                  'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                  'Update me on Supply Chain Content','Get updates on DM Content', 
                  'I agree to pay the amount through cheque','Tags'], axis=1)
df_leadscore4.head()

df_leadscore4.shape

## converting some binary variables (Yes/No) to 0/1

print(df_leadscore4["Do Not Email"].value_counts())
print(df_leadscore4["A free copy of Mastering The Interview"].value_counts())

# Replacing the values of yes =1 and No= 0
df_leadscore4["Do Not Email"].replace(to_replace = "No" , value = 0 , inplace = True)
df_leadscore4["Do Not Email"].replace(to_replace = "Yes" , value = 1 , inplace = True)
df_leadscore4["A free copy of Mastering The Interview"].replace(to_replace = "No" , value = 0 , inplace = True)
df_leadscore4["A free copy of Mastering The Interview"].replace(to_replace = "Yes" , value = 1 , inplace = True)

# checking the value of the data
df_leadscore4.head()

# 2b. Outlier Analysis

# Checking the data at 25%,50%,75%,90%,95% and above
df_leadscore4.describe(percentiles=[.25,.5,.75,.90,.95,.99])

* we can see clerar outlayers in TotalVisits and Page Views Per Visit columns, which need to be treated 

# Check the outliers for all the numeric columns

plt.figure(figsize=(25, 22))
plt.subplot(3,3,1)
sns.boxplot(y = 'TotalVisits', data = df_leadscore4)
plt.subplot(3,3,2)
sns.boxplot(y = 'Total Time Spent on Website', data = df_leadscore4)
plt.subplot(3,3,3)
sns.boxplot(y = 'Page Views Per Visit', data = df_leadscore4)
plt.show()

# Removing values beyond 99% for Total Visits

total_visits = df_leadscore4['TotalVisits'].quantile(0.99)
df_leadscore4 = df_leadscore4[df_leadscore4["TotalVisits"] < total_visits]
df_leadscore4["TotalVisits"].describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Removing values beyond 99% for page Views Per Visit

page_visits = df_leadscore4['Page Views Per Visit'].quantile(0.99)
df_leadscore4 = df_leadscore4[df_leadscore4["Page Views Per Visit"] < page_visits]
df_leadscore4["Page Views Per Visit"].describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Checking data again at 25%,50%,75%,90%,95% and above after removing values at 99 percentile
df_leadscore4.describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Finding the percentage of data retained

percent_data = round(100*(len(df_leadscore4)/9240),2)
print(percent_data)

* good to see that over all > 97% data is retained which is good to analisis 

## 3. Data Visualization :

#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))
# heatmap
sns.heatmap(df_leadscore4.corr(), cmap="Purples", annot=True)
plt.show()

df_leadscore4.columns

From the heat map, we do understand that, our continious cols are :

[ 'Do Not Email', 'Converted',
       'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
        'A free copy of Mastering The Interview']

So other cols are catagorical,
we already identified few as catagorical cols & named as cat_col 
others, can be named as cat_col1

cat_col1 = ['Lead Origin', 'Lead Source','Last Activity', 'What is your current occupation']

plt.figure(figsize = (15, 5))
for i in enumerate(cat_col1):
    plt.subplot(2,4,i[0]+1)
    print(i)
    sns.countplot(i[1], hue = 'Converted', data = df_leadscore4)
    plt.xticks(rotation = 90)

# Checking outliers for converted cols 

plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
sns.boxplot(df_leadscore4["Converted"] , df_leadscore4["Total Time Spent on Website"] , palette = "Blues_r")
plt.title("Total Time Spent on Website", fontdict={'fontsize': 20, 'color' : 'black'})

plt.subplot(1,2,2)
sns.boxplot(df_leadscore4["Converted"] , df_leadscore4["Page Views Per Visit"] , palette = "Blues_r")
plt.title("Page Views Per Visit", fontdict={'fontsize': 20, 'color' : 'brown'})

plt.show()

# Univariate Analysis - Categorical Variables

# Checking catagorical cols :

print(cat_cols)

#1. for col "Lead Source"

plt.figure(figsize = (20,7))
plt.subplots_adjust(hspace=0.2)

sns.countplot(df_leadscore4['Lead Source'], hue = df_leadscore4.Converted, palette = "PRGn")
plt.title('Lead Source', fontsize = 10, fontweight = 'bold')
plt.xticks(rotation = 45)

plt.show()

Remarks : Google has heighest convertion rate followed by 'Reference','Direct Traffic', 'Olark Chat' ...

#2. for col 'Lead Origin'

plt.figure(figsize = (20,7))
plt.subplots_adjust(hspace=0.2)

sns.countplot(df_leadscore4['Lead Origin'], hue = df_leadscore4.Converted, palette = "PRGn")
plt.title('Lead Source', fontsize = 10, fontweight = 'bold')
plt.xticks(rotation = 45)

plt.show()

Remarks : more students are endrolled with 'Landing Page Submission'

#3. for col 'Specialization'

plt.figure(figsize = (20,7))
plt.subplots_adjust(hspace=0.2)

sns.countplot(df_leadscore4['Specialization'], hue = df_leadscore4.Converted, palette = "PRGn")
plt.title('Lead Source', fontsize = 10, fontweight = 'bold')
plt.xticks(rotation = 45)

plt.show()

Remarks : 'Select'/ unknown cols have heighest lead rate 

#3. for col 'Last Notable Activity'

plt.figure(figsize = (20,7))
plt.subplots_adjust(hspace=0.2)

sns.countplot(df_leadscore4['Last Notable Activity'], hue = df_leadscore4.Converted, palette = "PRGn")
plt.title('Lead Source', fontsize = 10, fontweight = 'bold')
plt.xticks(rotation = 45)

plt.show()

Remarks: Students with 'Last Notable Activity' as 'SMS sent' is with heighest convertion rate 

#5. for col 'City'

plt.figure(figsize = (20,7))
plt.subplots_adjust(hspace=0.2)

sns.countplot(df_leadscore4['City'], hue = df_leadscore4.Converted, palette = "PRGn")
plt.title('Lead Source', fontsize = 10, fontweight = 'bold')
plt.xticks(rotation = 45)

plt.show()

Remarks : Studens from Mumbai have best convertion rate 

# Checking Data Imbalance

# Checking Imbalance of Data Converted_0 == 0(Lead not Converted) test_data1== 1 (Lead Converted)

Converted_0=df_leadscore4[df_leadscore4["Converted"]==0]
Converted_1=df_leadscore4[df_leadscore4["Converted"]==1]
print("Shape of All not Converted Leads -", Converted_0.shape)
print("Shape of All Converted Lead -", Converted_1.shape)


#Calculating Data Imbalance
imbalance= round((Converted_0.shape[0])/(Converted_1.shape[0]),3)
print("Imbalance Ratio is =",imbalance)


# ie, Converted_1 : Converted_0 = 1 : 1.597

# Checking data distribution using Pie Chart
plt.figure(figsize=[8,8])
plt.pie([Converted_0.shape[0],Converted_1.shape[0]], labels=["Not Converted Lead (0)","Converted Lead(1)"], autopct='%1.1f%%')
plt.title("Data Imbalance analysis\n", fontdict={'fontsize':20,'fontweight':6,'color':'blue'})
plt.show()

## 4. Data Preparation for Modelling

-Dummy creation 

-Splitting the Data into Training and Testing Sets

-Scaling The Features


# 4a. Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy = pd.get_dummies(df_leadscore4[['Lead Origin', 'What is your current occupation','City','Specialization','Lead Source', 'Last Activity', 'Last Notable Activity']], drop_first=True)

# Adding the results to the master dataframe
df_leadscore4 = pd.concat([df_leadscore4, dummy], axis=1)


# Dropping redandent cols 9as their dummy cols are included)
df_leadscore4=df_leadscore4.drop(['Lead Origin','What is your current occupation','City','Specialization','Lead Source', 'Last Activity', 'Last Notable Activity'],1)

# Checking current shape of data frame after dummy creation & redandent info elemination 
df_leadscore4.shape

df_leadscore4.head()

# Finding corelation matrix 
df_leadscore4.corr()


# 4b. Splitting the data into training and Splitting sets

# Putting feature variable to X

X = df_leadscore4.drop(['Converted'], axis = 1)
X.head()


y = df_leadscore4['Converted']
y.head()


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# Checking Shape of Train & Test
print("Train data shape :-",X_train.shape)
print("Test data shape :-",X_test.shape)

## Feature scaling using Minmax scaling

from sklearn.preprocessing import MinMaxScaler


# Scaling the three numeric features present in the dataset
scaler = MinMaxScaler()
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits',
                                                                                                              'Total Time Spent on Website',
                                                                                                              'Page Views Per Visit']])
X_train.head()


# Heatmap for correlation matrix
plt.figure(figsize=(20,30))
sns.heatmap(df_leadscore4.corr(), annot=True)
plt.show()

## 5.  Model Building

 - Course Tuning : top "n" feature selection (by RFE)
 - Fine tunning : 

import statsmodels.api as sm

# Logistic regression model with StatsModels
X_train=sm.add_constant(X_train)



#X_train = np.asarray(X_train, dtype=np.float64)

lgm1=sm.GLM(y_train,X_train, family=sm.families.Binomial())


lgm1.fit().summary()

Course selection : Now we are good to select the top "n" features 

## Top "n" selection using RFE 


#RFE - Selecting 15 Variables using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# Top 15 Features Selected by RFE for Modelling are:
rfe.support_


# Check all cols - which are supporting RFE or Not based on their ranking 

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

## Lets c, kaun hay wo bhagyasali vijeta !! - kidding, lets c what are the selected top 15 features 
col = X_train.columns[rfe.support_]
col


# Lets check Dataset of columns, selected by RFE:
X_train[col].head()

## 5b Fine tuning - Building Models 

# BUILDING MODEL-1
# Fit a Logistic Regression Model on X-train after adding a constant and output the summary 

# Adding constant
X_train_sm = sm.add_constant(X_train[col])

# Running the model
lm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

# Fit a line
res = lm1.fit()

# Checking the model summary
res.summary()

Remarks:Now we meed to get look upon the p-value & VIF score, to eleminate the redundat features (ie, features having stong co-relation with eeach other), & finally will start to eleminate 1 by one to get the final model such that all features will have "0" p-value & VIF <3

# Checking the VIF for the new model-
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# dropping column with high p-value
col = col.drop('What is your current occupation_Housewife',1)

# Building Model 2
X_train_sm = sm.add_constant(X_train[col])
lm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = lm2.fit()
res.summary()

# Checking the VIF for the model2-
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# dropping column with high p-value
col = col.drop('Last Notable Activity_Had a Phone Conversation',1)

# Building Model 3
X_train_sm = sm.add_constant(X_train[col])
lm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = lm2.fit()
res.summary()

# Checking the VIF for the model3-
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

from VIF value, we now get the feature "What is your current occupation_Unemployed" is having VIF>3 , means it might have a strong coreation with other variable , hence eleminating this feature

Now we have our final model. The p values represents significance of the variables and VIF represent correlation of variable with each other. The VIFs and p-values both are within an acceptable range. So we will move further and make our predictions using this model only.

# Checking & Verifying & Validating Correlations Again:
plt.figure(figsize=(15,10), dpi=80, facecolor='w', edgecolor='k', frameon='True')

corr = X_train[col].corr()
sns.heatmap(corr, annot=True, cmap="PuBu")

plt.tight_layout()
plt.show()

# 6. Model Prediction :

# Predicting the probabilities (of the "converted" value being 1) on the train set
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]

#Creating a dataframe with the actual converted flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Prospect Id'] = y_train.index
y_train_pred_final.head()

Remarks : To test/ evalute our predictive model, lets create an col "Pridicted" based on the Converted probability (ie, "Converted_Prob"), such that if Converted_Prob >0.5 (ie, more than 50% chances for convertion), "Predicted" will be 1, else 0. By this way we can classify  it as 'Not Converted'or 'Converted' 

#Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()

# 8. Model Evaluation

# Checking the confusion metrics
confusion = confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)

# checking the overall accuracy
print(accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Checking the sensitivity of our logistic regression model
TP / float(TP+FN)

# Let us calculate specificity
TN / float(TN+FP)

# Calculate false postive rate - predicting non conversion when leads have converted
print(FP/ float(TN+FP))

# True positive predictive value 
print (TP / float(TP+FP))

# Negative predictive value
print (TN / float(TN+ FN))

Remarks : Our model seems to have high accuracy (80.99%), high sensitivity (68.80%) and high specificity (88.5%). We will indentify the customers which might convert, with the help of ROC Curves in the next section.

# 8. Plotting ROC Curve

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

# TPR = True Positive Rate --> Out of actually 'converted' cases, how many is the model correctly predicting as 'converted'
# FPR = False Positive Rate --> Out of actual 'non-convert' cases, how many is the model incorrectly predicting as 'converted'

# Thus we would want as high TPR as possible & as low FPR as possible. Ideal model would have TPR = 1 and FPR = 0.
# ROC Curve shows the trade-off between TPR and FPR.

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)

Remarks : The area under the curve of the ROC is nearly equal to 1 (precisely 0.88) which is quite good. so we seem to have a good model. Lets also check the sensitivity and specificity tradeoff to find the optimal cut off point.

# 9. Finding Optimal Cutoff Point

# create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_p = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_p.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_p)

#Plotting Graph
# Let's plot accuracy sensitivity and specificity for various probabilities.
sns.set_style("whitegrid") 
sns.set_context("paper") 
cutoff_p.plot.line(x='prob', y=['accuracy','sensi','speci'], figsize=(13,6))
# plot x axis limits
plt.xticks(np.arange(0, 1, step=0.05), size = 12)
plt.yticks(size = 12)
plt.show()

# Evaluating the model with optimal probability cutoff as 0.35
y_train_pred_final['final_Predicted']=y_train_pred_final.Converted_Prob.map(lambda x: 1 if x>0.35 else 0)
y_train_pred_final.head()

# Now let us calculate the lead score

y_train_pred_final['lead_score_board'] = y_train_pred_final.Converted_Prob.map(lambda x: round(x*100))
y_train_pred_final[['Converted','Converted_Prob','Prospect Id','final_Predicted','lead_score_board']].head(10)

#Lets check  accuracy
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted))

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

# Sensitivity 
TP / float(TP+FN)

# specificity
TN / float(TN+FP)

# false postive rate
print(FP/ float(TN+FP))

# True Positive predictive value 
print (TP / float(TP+FP))

# True Negative predictive value
print (TN / float(TN+ FN))

Accuracy : 88.32%
Sensitivity : 86.63%
Specificity : 88.74%

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)
confusion

# Precision:
pre= TP/ (TP + FP)
pre

# Recall:
rec=TP/ (TP + FN)
rec

from sklearn.metrics import classification_report

print(classification_report(y_train_pred_final["Converted"],y_train_pred_final["final_Predicted"]))

F1 = 2*(pre*rec)/(pre+rec)
F1

y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)

plt.figure(figsize=[7,5])
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

# Making Predictions on test set

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits',
                                                                        'Total Time Spent on Website','Page Views Per Visit']])


X_test = X_test[col]
X_test.head()

print(y_test.shape)
print(X_test.shape)

#Add constant to X_test
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]

# Converting y_pred to a dataframe which is an array
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.head()

# Converting y_test to dataframe
y_test_dataframe = pd.DataFrame(y_test)

y_test_dataframe['Prospect Id'] = y_test_dataframe.index

# Removing index for both dataframes to append them side by side 
y_test_pred.reset_index(drop=True, inplace=True)
y_test_dataframe.reset_index(drop=True, inplace=True)

# Appending y_test_dataframe and y_test_pred
y_pred_final = pd.concat([y_test_dataframe, y_test_pred],axis=1)

# Checking the head() of the final dataframe
y_pred_final.head()

# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

y_pred_final['Lead_Score_Board'] = y_pred_final.Converted_Prob.map(lambda x: round(x*100))
y_pred_final.head()

# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Prospect Id','Converted','Converted_Prob', 'Lead_Score_Board'], axis=1)

# Let's see the head of y_pred_final
y_pred_final.head()

y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()

## Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted)
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)

# Let us calculate specificity
TN / float(TN+FP)

Accuracy: 80.36%

Sensitivity: 81.70%

Specificity: 79.48%

# precision
precision_score(y_pred_final.Converted , y_pred_final.final_predicted)

#recall
recall_score(y_pred_final.Converted, y_pred_final.final_predicted)

# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))

# True Positive predictive value 
print (TP / float(TP+FP))

# True Negative predictive value
print (TN / float(TN+ FN))

print(classification_report(y_pred_final["Converted"],y_pred_final["final_predicted"]))

# Precision and Recall metrics for the test set

# precision
print('Precision - ',precision_score(y_pred_final.Converted, y_pred_final.final_predicted))

# recall
print('Recall -',recall_score(y_pred_final.Converted, y_pred_final.final_predicted))

p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Converted_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

Remarks : To avoid overfitting, let us calculate the Cross Validation Score to see how our model performs


from sklearn.model_selection import cross_val_score

X=X_train[:200]
y=y_train[:200]

lr = LogisticRegression(solver = 'lbfgs')
scores = cross_val_score(lr, X, y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)

# ROC Curve 
fpr, tpr, thresholds = metrics.roc_curve(y_pred_final["Converted"], y_pred_final["Converted_Prob"], drop_intermediate = False )
draw_roc(y_pred_final["Converted"], y_pred_final["Converted_Prob"])

# Calculating the LEAD SCORE

#This needs to be calculated for all the leads from the original dataset (train + test)
leads_test_pred = y_pred_final.copy()
leads_test_pred.head()

# Selecting the train dataset along with the Conversion Probability and final predicted value for 'Converted'
leads_train_pred = y_train_pred_final.copy()
leads_train_pred.head()

# Dropping unnecessary columns from train dataset
leads_train_pred = leads_train_pred[['Prospect Id','Converted','Converted_Prob','final_Predicted']]
leads_train_pred.head()

leads_test_pred = leads_test_pred[['Prospect Id','Converted','Converted_Prob','final_predicted']]
leads_test_pred.head()

# Concatenating the 2 dataframes train and test along the rows with the append() function
lead_full_pred = leads_train_pred.append(leads_test_pred)
lead_full_pred.head()

# Calculating the Lead Score value
# Lead Score = 100 * Conversion_Prob
lead_full_pred['Lead_Score'] = lead_full_pred['Converted_Prob'].apply(lambda x : round(x*100))
lead_full_pred.head()

# Inspecting the dataframe shape
lead_full_pred.shape

# Making the Prospect ID column as index

lead_full_pred = lead_full_pred.set_index('Prospect Id').sort_index(axis = 0, ascending = True)
lead_full_pred.head()

# Determining HOT LEADS with 81% accuracy more than 80% Conversion Rate

# Determining hot leads with more than 80% Conversion Rate
hot_leads = lead_full_pred[lead_full_pred["Lead_Score"]>80]
hot_leads.head()

# Hot Leads Shape
hot_leads.shape

# Determining Feature Importance

# Selecting the coefficients of the selected features from our final model excluding the intercept

pd.options.display.float_format = '{:.2f}'.format
new_params = res.params[1:]
new_params

# Getting a relative coeffient value for all the features wrt the feature with the highest coefficient

feature_importance = new_params
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance.sort_values(ascending = False)

# Ranking features based on importance

# To sort features based on importance
sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')
sorted_idx

# Plot showing the feature variables based on their relative coefficient values
plt.figure(figsize = (15,10))
feature_importance.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature variables based on their relative coefficient')
plt.ylabel('Relative Coefficient')
plt.show()

# Selecting Top 3 features which contribute most towards the probability of a lead getting converted

pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False).head(3)

# Selecting Least 3 features which contribute most towards the probability of a lead getting converted

pd.DataFrame(feature_importance).reset_index().sort_values(by=0,ascending=False).tail(3)

RESULTS
Final Observation:

So as we can see above the model seems to be performing well. The ROC curve has a value of 0.88, which is very good. Let us compare the values obtained for Train & Test Set:

Train Data:
Accuracy: 80.99%
Sensitivity: 68.80%
Specificity: 88.51%


Test Data:
Accuracy: 80.36%
Sensitivity: 81.70%
Specificity: 79.48%

# CONCLUSION & RECOMMENDATIONS
After trying several models, we finally chose a model no 5 with the following characteristics:
All variables have p-value < 0.05, showing significant features contributing towards Lead Conversion.

All the features have very low VIF values (<3), means hardly there is any muliticollinearity among the features. This can be seen from the heat map.

The ROC curve has a value of 0.88, which is very good!

The overall accuracy of Around 89% at a probability threshold of 0.34 on the test dataset is also very acceptable.

For Train Dataset

Accuracy: 80.99%
Sensitivity: 68.80%
Specificity: 88.51%
False postive rate - predicting the lead conversion when the lead does not convert: 0.1148
Precision/Positive predictive value: 78.70%
Negative predictive value: 82.14%
ROC : 0.95
F1 Score : 0.76

For Test Dataset

Accuracy: 80.36%
Sensitivity: 81.70%
Specificity: 79.48%
pecificity: 79.48%
Precision -  0.7210264900662252
Recall - 0.8170731707317073
FP = 0.20511259890444308
TP = 0.7210264900662252
TN = 0.7210264900662252
ROC : 0.88
The optimal threshold for the model is 0.35 which is calculated based on tradeoff between sensitivity, specificity and accuracy. According to business needs, this threshold can be changed to increase or decrease a specific metric.

High sensitivity ensures that most of the leads who are likely to convert are correctly predicted, while high specificity ensures that most of the leads who are not likely to convert are correctly predicted.

THANK YOU
Submitted by Neelanjan Roy | Akshay Roy | Sameer Kumar : DSC 54 batch
Neelanjan Roy: roy.neelanjan@gmail.com
Akshay Roy :  royakshay46@gmail.com
Sameer Kumar: rsameer.kumar17@gmail.com


