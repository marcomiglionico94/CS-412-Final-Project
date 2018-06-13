
# coding: utf-8

# In[329]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[330]:

df = pd.read_csv('lending-club-loan-data/loan.csv', low_memory=False)


# In[331]:

df.info()


# In[396]:

pd.set_option('display.max_columns', None) 
df_clean.head()


# In[333]:

df_clean.info()


# # Feature selection

# ##### The focus of the project is to detect if a loan will be fully paid or charged off based on the information available at the moment the user is requesting the loan

# In[334]:

df_clean = df[(df.loan_status == "Fully Paid") | (df.loan_status == "Charged Off")]


# ##### Remove joint applications since are there are few instances of them in the dataset and in this way we can remove the set of features available only for joint applications

# In[335]:

df.application_type.value_counts()


# In[336]:

df_clean = df_clean[df.application_type == "INDIVIDUAL"]
df_clean = df_clean.drop("application_type", axis=1)


# In[337]:

df_clean = df_clean.drop(['annual_inc_joint', 'dti_joint', 'verification_status_joint'], axis = 1)


# ##### Remove ID features

# In[338]:

df_clean = df_clean.drop(['id'], axis = 1)
df_clean = df_clean.drop(['member_id'], axis = 1)


# ##### Remove job title: non informative

# In[339]:

df_clean = df_clean.drop(['emp_title'], axis = 1)


# ##### Remove grade: Info already available in subgrade

# In[340]:

df_clean = df_clean.drop(['grade'], axis = 1)


# ##### Remove URL, description and title: Descriptive information not useful

# In[341]:

df_clean = df_clean.drop(['url', 'desc', 'title'], axis = 1)


# ##### Remove policy code, out_prncp and out_prncp_inv since they are all the same in the clean dataset

# In[342]:

df_clean.policy_code.value_counts()


# In[343]:

df_clean.out_prncp.value_counts()


# In[344]:

df_clean.out_prncp_inv.value_counts()


# In[345]:

df_clean = df_clean.drop(['policy_code', 'out_prncp', 'out_prncp_inv'], axis = 1)


# ##### Remove features not strictly related to application details

# In[346]:

df_clean = df_clean.drop(['total_pymnt', 'total_pymnt_inv'], axis = 1)


# In[347]:

df_clean = df_clean.drop(['total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'], axis = 1)


# In[348]:

df_clean = df_clean.drop(['recoveries'], axis = 1)


# In[349]:

df_clean = df_clean.drop([
    'collection_recovery_fee',
    'last_pymnt_d',
    'last_pymnt_amnt',
    'last_credit_pull_d',
    'collections_12_mths_ex_med'
], axis = 1)


# In[350]:

df_clean = df_clean.drop(['acc_now_delinq'], axis = 1)


# In[351]:

df_clean = df_clean.drop(['tot_cur_bal'], axis = 1)


# ##### Remove features with too many missing entries and no reason to impute them

# In[352]:

df_clean.next_pymnt_d.value_counts()


# In[353]:

df_clean = df_clean.drop(['next_pymnt_d'], axis = 1)


# In[354]:

df_clean = df_clean.drop([
    'open_acc_6m',
    'open_il_6m',
    'open_il_12m',
    'open_il_24m',
    'mths_since_rcnt_il',
    'total_bal_il',
    'il_util',
    'open_rv_12m',
    'open_rv_24m',
    'max_bal_bc',
    'all_util',
    'inq_fi',
    'total_cu_tl',
    'inq_last_12m',
], axis = 1)


# In[355]:

df_clean = df_clean.drop([
    'tot_coll_amt',
    'total_rev_hi_lim',
], axis = 1)


# ##### mths_since_last_major_derog: Months since most recent 90-day or worse rating
# We assume that if the value is null, it means that no 90-day or worse ratings have ever been appointed to the applicant and so a reasonable high value is set.

# In[356]:

df_clean.mths_since_last_major_derog.describe()


# In[357]:

fig, ax = plt.subplots()
fig.set_size_inches(90.7, 8.27)
sns.countplot(x='mths_since_last_major_derog', data=df_clean)


# In[358]:

df_clean.mths_since_last_major_derog.head()


# In[359]:

df_clean["mths_since_last_major_derog"] = df_clean["mths_since_last_major_derog"].apply(lambda x: 500.0 if pd.isnull(x) else x)


# ##### mths_since_last_delinq: The number of months since the borrower's last delinquency.
# We assume that if the value is null, it means that no delinquency has ever been appointed to the applicant and so a reasonable high value is set.

# In[360]:

df_clean["mths_since_last_delinq"] = df_clean["mths_since_last_delinq"].apply(lambda x: 500.0 if pd.isnull(x) else x)


# ##### mths_since_last_record: The number of months since the last public record.
# We assume that if the value is null, it means that no public record has ever been appointed to the applicant and so a reasonable high value is set.

# In[361]:

df_clean["mths_since_last_record"] = df_clean["mths_since_last_record"].apply(lambda x: 500.0 if pd.isnull(x) else x)


# ##### Remove funded_amnt_inv
# We're interested in funded_amnt

# In[362]:

df_clean = df_clean.drop(['funded_amnt_inv'], axis = 1)


# ##### Remove payment plan: not informative, only 2 entries with value different from the others
# We're interested in funded_amnt

# In[363]:

df_clean = df_clean.drop(['pymnt_plan'], axis = 1)


# ##### Remove sub_grade
# mapped to int_rate

# In[364]:

df_clean = df_clean.drop(['sub_grade'], axis = 1)


# ### Remove instances with missing values

# In[365]:

df_clean = df_clean.dropna(subset = ['revol_util'])


# ### Remove instances that differs between loan_amnt and funded_amnt to reduce noise

# In[366]:

df_temp = df_clean[(df_clean.loan_amnt != df_clean.funded_amnt)]


# In[367]:

df_temp["loan_status"].value_counts()


# In[368]:

df_clean.loan_status.value_counts()


# In[369]:

df_clean = df_clean.drop(df_clean[df_clean.loan_amnt != df_clean.funded_amnt].index)


# In[370]:

df_clean = df_clean.drop(['funded_amnt'], axis = 1)


# # Data preparation

# #### Remove "months" from term

# In[371]:

df_clean["term"] = df_clean["term"].str.split(" ").str[1]
df_clean["term"] = df_clean["term"].astype(float)


# #### Managing issuing and credit history date
# 
# **issue_d**: The date which the borrower accepted the offer
# 
# **earliest_cr_line**: The date the borrower's earliest reported credit line was opened
# 
# Delete this two feature and add a new one (last_first_credit_diff) which is the difference in number of months beteween the earliest reported credit line opened and the date in which the considered loan has been issued.

# In[372]:

df_clean.issue_d.head()


# In[373]:

df_clean.earliest_cr_line.head()


# In[374]:

m = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# In[375]:

df_clean["earliest_cr_line"] = df_clean["earliest_cr_line"].apply(lambda x: m[x.split("-")[0]] + int(x.split("-")[1])*12)
df_clean["issue_d"] = df_clean["issue_d"].apply(lambda x: m[x.split("-")[0]] + int(x.split("-")[1])*12)


# In[376]:

df_clean["last_first_credit_diff"] = df_clean["issue_d"] - df_clean["earliest_cr_line"]


# In[377]:

df_clean = df_clean.drop(['issue_d', 'earliest_cr_line'], axis = 1)


# #### initial_list_status
# categorized values

# In[378]:

df_clean["initial_list_status"] = df_clean["initial_list_status"].astype('category')
df_clean["initial_list_status"] = df_clean["initial_list_status"].cat.codes


# #### home_ownership
# Remove unique instance with value ANY and categorized values

# In[379]:

df_clean.home_ownership.value_counts()


# In[380]:

df_clean = df_clean.drop(df_clean[df_clean.home_ownership == "ANY"].index)


# In[381]:

df_clean.home_ownership.value_counts()


# In[382]:

df_clean["home_ownership"] = df_clean["home_ownership"].astype('category')
df_clean["home_ownership"] = df_clean["home_ownership"].cat.codes


# In[383]:

df_clean.home_ownership.value_counts()


# #### verification_status
# categorized values

# In[384]:

df_clean["verification_status"] = df_clean["verification_status"].astype('category')
df_clean["verification_status"] = df_clean["verification_status"].cat.codes


# #### purpose
# categorized values

# In[385]:

df_clean["purpose"] = df_clean["purpose"].astype('category')
df_clean["purpose"] = df_clean["purpose"].cat.codes


# In[386]:

df_clean.purpose.value_counts()


# #### emp_length

# In[387]:

df_clean.emp_length.value_counts()


# In[388]:

emps = {
    "10+ years": 15.0,
    "2 years": 2.0,
    "< 1 year": 0.5,
    "3 years": 3.0,
    "5 years": 5.0,
    "1 year": 1.0,
    "4 years": 4.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "n/a": 0.0,
    "9 years": 9.0,
}


# In[389]:

df_clean["emp_length"] = df_clean["emp_length"].apply(lambda x: emps[x])


# #### Geographical deatures
# Considered only first two digits of the zipcode
# 
# Categorized addr_state

# In[390]:

df_clean["zip_code"].describe()


# In[391]:

df_clean["zip_code"] = df_clean["zip_code"].apply(lambda x: int(x[:2]))


# In[392]:

df_clean["addr_state"].describe()


# In[393]:

df_clean["addr_state"] = df_clean["addr_state"].astype('category')
df_clean["addr_state"] = df_clean["addr_state"].cat.codes


# #### Categorized label
# fully paid = 0, charged off = 1

# In[394]:

df_clean["loan_status"] = df_clean["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)


# In[395]:

df_clean["loan_status"].value_counts()


# # Export

# In[397]:

df_clean.to_csv("loan_clean.csv", sep=',')
