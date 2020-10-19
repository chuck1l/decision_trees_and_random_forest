import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use('seaborn')

loans = pd.read_csv('../data/loan_data.csv')
toggle = False  
if toggle:
    print(loans.head()) 
    print(loans.info())
    print(loans.describe())
# Exploritory Data Analysis
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.title('FICO Score Distributions, Per Credit Policy')
plt.xlabel('FICO')
plt.ylabel('Count')
#plt.savefig('../imgs/fico_dist_per_policy.png')
plt.show();

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.title('FICO Score Distributions, Per Fully Paid or Not')
plt.xlabel('FICO')
plt.ylabel('Count')
#plt.savefig('../imgs/not_paid_by_fico_score.png')
plt.show();

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
plt.title('Fully Paid or Not, By Purpose')
#plt.savefig('../imgs/purpose_not_paid_infull.png')
plt.show();

sns.jointplot(x='fico',y='int.rate',data=loans, color='purple')
#plt.savefig('../imgs/scatter_fico_vs_intrate.png')
plt.show();

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
#plt.savefig('../imgs/trend_differ_credipolicy_rate_fico.png')
plt.show();

plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='not.fully.paid',
           col='credit.policy',palette='Set1')
plt.savefig('../imgs/trend_differ_notpaid_rate_fico.png')
plt.show();