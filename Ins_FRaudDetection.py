#Importing libraries

import pandas as pd


pd.set_option('display.max_columns', None)
#%%Importing dataset
claims = pd.read_csv("D:\clg shit\Sem 4\pyt\Prj\healthcare_fraud_detection.csv")

#%%line 13 to 16 fill in nan values using mode and mean of the data,I did not drop them to keep all data points, the nan values equated to around ~3.5% of all values

claims['Insurance_Type']=claims['Insurance_Type'].fillna(claims['Insurance_Type'].mode()[0])
claims['Provider_Specialty']=claims['Provider_Specialty'].fillna(claims['Provider_Specialty'].mode()[0])
claims['Prior_Visits_12m']=claims['Prior_Visits_12m'].fillna(claims['Prior_Visits_12m'].mean())
print(claims.isnull().sum())


#%% Excluding Self-Pay from insurance type since it is not a type of insurance and has nothing to do it modelling for fraud detection
print(claims['Insurance_Type'].value_counts())
print(claims[claims['Insurance_Type']=='Self-Pay']['Is_Fraud'].mean())
claims= claims[claims['Insurance_Type']!= 'Self-Pay']


# %%POR for all columns that show high corr to fraud(Very Important)
 
print("                                       <NUMERIC>                                        ")
cols=['Claim_Amount', 'Approved_Amount','Number_of_Claims_Per_Provider_Monthly','Days_Between_Service_and_Claim', 'Prior_Visits_12m','Chronic_Condition_Flag' ]

No=claims[claims['Is_Fraud'] == 0][cols].mean()
Yes= claims[claims['Is_Fraud']== 1][cols].mean()

summary = pd.DataFrame({'Not fraud':No, 'Fraud':Yes})
summary['Driffrence %']=((summary['Fraud']-summary['Not fraud'])/summary['Not fraud']*100).round(1)

print(summary)
print("                                     <NON_NUMERIC>                                      ")


for col in ['Insurance_Type', 'Visit_Type','Provider_Specialty']:
    print(((claims.groupby(col)['Is_Fraud'].mean())*100))
    print()


print("Observations:                                                                               1.All non numeric colums have almost identical(<10%) diffrence                                                           2. Days between service and claim has a high -ve correlation i.e fraud claims are being filed quicker")

# %% Model to detect fraud 

Features=['Claim_Amount', 'Approved_Amount','Number_of_Claims_Per_Provider_Monthly','Days_Between_Service_and_Claim']

from sklearn.model_selection import train_test_split
X= claims[Features]
Y = claims['Is_Fraud']


X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=72)

#%% checking the data for imbalance before modelling


print(claims['Is_Fraud'].value_counts())
print(claims['Is_Fraud'].value_counts(normalize=True))

#%% model

from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(class_weight= 'balanced', random_state=72)
model.fit(X_train,Y_train)


#%% Evaluation

from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,precision_score,f1_score
y_pred= model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print("Accuracy:",accuracy_score(Y_test, y_pred))
print("Recall score:",recall_score(Y_test, y_pred))
print("Precision:",precision_score(Y_test, y_pred))
print("F1 score:",f1_score(Y_test, y_pred))

AUC= roc_auc_score(Y_test, y_prob)
print("ROC:",AUC)

importance = pd.Series(model.feature_importances_, index=Features)
print(importance.sort_values(ascending=False))

#%% ROC curve plot
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(Y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {AUC:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Healthcare Fraud Detection")
plt.legend()
plt.show()
