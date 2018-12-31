# Import Libraries

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn import  linear_model
from sklearn.ensemble import IsolationForest
import matplotlib as mpl
import os
from sklearn import metrics
import datetime
import sqlalchemy as sa
from sqlalchemy.engine import url as sa_url 
from sqlalchemy import create_engine
import scipy.stats.mstats as spy
from sklearn import svm

# declare variables and global parameters

output_table = pd.DataFrame(columns=['RU Milestone', 
                                     'RU', 'CP Amount',
                                     'CP Predicted',
                                     'Milestone', 
                                     'RU Description', 
                                     'Variance Upto Last 6 months',
                                     'Variance Last 6 months', 
                                     'CP Z-Score',
                                     'Gradient Upto Last 6 months',
                                     'Intercept Upto Last 6 months',
                                     'Gradient Last 6 months',
                                      'Intercept Last 6 months',
                                      'Transaction Volumes'
                                     ])
detailed_results = pd.DataFrame()
output_table_detailed = pd.DataFrame()

confidence_level = 1.96
materiality = 100
rng = np.random.RandomState(42)



# Connect to database
# Note:  pip install pymssql

sql_engine = sa.create_engine('mssql+pymssql://reporting:Summ3r123@AUSYDAPD38/Finance_BPO')

with sql_engine.connect() as conn, conn.begin():
    df = pd.read_sql("""SELECT
	"posting_month_id" as "Period",
	"reporting_unit" as "Reporting Unit",
	"reporting_unit_description" as "RU Description",
	"Milestone",
	MAX("full_work_day") AS "Full Work Day",
	COUNT("reporting_unit") AS "Transaction volumes",
	"reporting_unit" + ' ' + "Milestone" as "Key"
FROM(
	SELECT
	  CONVERT(datetime,LEFT(posting_date_id,6)+'01') AS "posting_month_id",
	  f."full_work_day_controllership_time" AS "full_work_day",
	  r."reporting_unit" as "reporting_unit",
	  r."reporting_unit_description" as "reporting_unit_description",
	  CASE 
		   WHEN activity = 'SD00' THEN 'Underlying Sales'
		   WHEN taxonomy_category = 'Journal' and "rtgc_account_category_1" = 'Revenue & Sales Expenses'THEN 'Sales Journal'
		   WHEN order_number LIKE '____9%' AND taxonomy_category in ('Production Confirmation','General Materials') THEN 'Production Confirmation'
		   WHEN (account LIKE '16____' OR account LIKE '17____') AND account NOT LIKE '_____5' AND USER_NAME LIKE 'ECP%' THEN 'Banking Interface'
		   WHEN (account LIKE '16____' OR account LIKE '17____') AND account NOT LIKE '_____5' AND USER_NAME NOT LIKE 'ECP%' THEN 'Banking Manual'
		   WHEN document_type in ('Y1', 'Y2', 'YI', 'YJ') THEN 'Intercompany Recharge'
		   WHEN taxonomy_category = 'Journal' THEN 'Other Journal'
		   WHEN taxonomy_category = 'P&L Jobs' THEN 'P&L Jobs'
		   WHEN taxonomy_category = 'Balance sheet jobs' THEN 'Balance sheet jobs'
		   WHEN taxonomy_category = 'HFM Adjustment' THEN 'HFM Adjustment'
		   WHEN taxonomy_category = 'HFM Journal' THEN 'HFM Journal'
	   
		   ELSE 'Other'
		   END AS Milestone


	FROM "rdl"."t_fact_monthend_activity_detail" f
	  INNER JOIN "rdl"."t_dim_date" "posting_date" ON (f."posting_date_id" = "posting_date"."date_id")
	  INNER JOIN rdl.t_dim_taxonomy t on f.taxonomy_id_accounting = t.taxonomy_id
	  INNER JOIN rdl.t_dim_account_category a on f.account_id =  a.account_id
	  INNER JOIN rdl.t_dim_reporting_unit_details r on f.reporting_unit_id =  r.reporting_unit_id
	WHERE
	  "posting_date"."relative_period" BETWEEN -24 AND -1
	  AND work_day_controllership BETWEEN -0 and 7
	  ) X

WHERE
	"full_work_day" IS NOT NULL
    AND reporting_unit IN ('2190','2030','2600','2240','2510','224003','224004','224005','224008', '226401')
---    AND reporting_unit = '226401'
GROUP BY 
	"posting_month_id",
	"reporting_unit",
	"reporting_unit_description",
	"Milestone"
    """, conn)
    
# Additional transforms

df['Months Since Start'] = (datetime.datetime.now() - df['Period']).astype('timedelta64[M]')


# Linear Regression

list_of_keys = df['Key'].drop_duplicates()

for I in list_of_keys:

    
    # acquire data
    
    keys_filter = df['Key'] == I
    filtered_data = df.loc[keys_filter]
    filtered_data = filtered_data.sort_values(by=['Months Since Start'], ascending=False)

    if len(filtered_data['Full Work Day']) > 12:
    
#       sorted_data = filtered_data.sort_values('Full Work Day')
        X_full = filtered_data[['Months Since Start']].values
        y_full = filtered_data['Full Work Day'].values

        
        # isolation forest to detect anomalies

        y_full_reshape = y_full.reshape(-1,1)

        clf = IsolationForest(random_state=rng)
        clf.fit(y_full_reshape)
        clf_predicted_amount = clf.predict(y_full_reshape)      
        
        anomaly_test_false = clf_predicted_amount == 1
        anomaly_test_true = clf_predicted_amount == -1

        X = X_full[anomaly_test_false]
        y = y_full[anomaly_test_false]

        
        X_upto_last_6months = X[0:-6]
        y_upto_last_6months = y[0:-6]
        X_last_6months = X[-6:]
        y_last_6months = y[-6:]


# run regression
       
        # Upto last 6 months

        
        reg = linear_model.LinearRegression()
        reg.fit(X_upto_last_6months,y_upto_last_6months)
        predicted_amount_upto_last_6months_MAE = reg.predict(X_upto_last_6months)
        predicted_amount_upto_last_6months = reg.predict(X_full[0:-6])
    
            # Last 6 months
    
        reg6months = linear_model.LinearRegression()
        reg6months.fit(X_last_6months,y_last_6months)
        predicted_amount_last_6months_MAE = reg.predict(X_last_6months)
        predicted_amount_last_6months = reg6months.predict(X_full[-6:])
    
    
    
        # calculate metrics

        # before last 6 months    
        
        MAE_upto_6months = metrics.mean_absolute_error(y_upto_last_6months, predicted_amount_upto_last_6months_MAE, sample_weight=None)
        r_squared_upto_6months = metrics.r2_score(y_full[0:-6], predicted_amount_upto_last_6months, sample_weight=None)
        upper_bound_upto_last_6months = MAE_upto_6months*confidence_level + predicted_amount_upto_last_6months
        lower_bound_upto_last_6months = MAE_upto_6months*-confidence_level + predicted_amount_upto_last_6months
        confidence_range_upto_last_6months = upper_bound_upto_last_6months - lower_bound_upto_last_6months

        # last 6 months    
        
        MAE_last_6months = metrics.mean_absolute_error(y_last_6months, predicted_amount_last_6months_MAE, sample_weight=None)
        r_squared_last_6months = metrics.r2_score(y_full[-6:], predicted_amount_last_6months, sample_weight=None)
        upper_bound_last_6months = MAE_last_6months*confidence_level + predicted_amount_last_6months
        lower_bound_last_6months = MAE_last_6months*-confidence_level + predicted_amount_last_6months
        confidence_range_last_6months = upper_bound_last_6months - lower_bound_last_6months


       
        # construct detailed output file
        

        
        detailed_results = pd.DataFrame()
        
        detailed_results['Full Work Day'] = y_full
        detailed_results['Predicted'] = np.concatenate((predicted_amount_upto_last_6months, predicted_amount_last_6months), axis=None)
        detailed_results['Lower bound'] = np.concatenate((lower_bound_upto_last_6months, lower_bound_last_6months), axis = None)
        detailed_results['Upper bound'] = np.concatenate((upper_bound_upto_last_6months, upper_bound_last_6months), axis = None)
        detailed_results['Confidence range'] = np.concatenate((confidence_range_upto_last_6months, confidence_range_last_6months), axis = None)
        detailed_results['RU Milestone'] = I
        detailed_results['Reporting Unit'] = filtered_data.iloc[0,1]
        detailed_results['RU Description'] = filtered_data.iloc[0,2]
        detailed_results['Milestone'] = filtered_data.iloc[0,3]
        detailed_results['Transaction Volumes'] = filtered_data.iloc[0,5]
        detailed_results['Months since start'] = filtered_data['Months Since Start'].values
        detailed_results['Period'] = filtered_data['Period'].values
        detailed_results['Anomaly'] = anomaly_test_true
        
        
        output_table_detailed = output_table_detailed.append(detailed_results)
    
        # Chart anomalies
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.style.use('seaborn-white')
        plt.scatter(filtered_data['Months Since Start'][anomaly_test_false],y_full[anomaly_test_false], color='blue')
        plt.scatter(filtered_data['Months Since Start'][anomaly_test_true],y_full[anomaly_test_true], color='red')
        plt.plot(filtered_data['Months Since Start'],detailed_results['Predicted'], color='black', linewidth=2)
        plt.fill_between(filtered_data['Months Since Start'],detailed_results['Lower bound'],  detailed_results['Upper bound'],color = '#539caf', alpha = 0.4)
        plt.yscale('linear')
        plt.legend(('Predicted', 'Normal Result', 'Anomaly Result'))
        plt.title(filtered_data.iloc[1,3]+" - "+filtered_data.iloc[1,1])
        plt.xlim(26,0)
        plt.xlabel('Months before current period')
        plt.show()
        


        #print(filtered_data.iloc[1,3]+" - "+filtered_data.iloc[1,1])
        #print('Mean absolute error - ')
        # print(MAE)
        #print('R-squared - ') 
        #print(r_squared)
        #print ('Z_score')
        #print(z_score)
        #print('Co-efficients')
        #print (reg.coef_)

        # Key RU / Milestone measures

        current_period_amount = y_full[-1]
        current_period_predicted = detailed_results.iloc[-1,1]
        z_score = (current_period_amount - current_period_predicted) / MAE_last_6months
        gradient_last_6_months = reg.coef_
        intercept_last_6_months = reg.intercept_
        gradient_upto_last_6_months = reg6months.coef_
        intercept_upto_last_6_months = reg6months.intercept_
        transaction_volumes = detailed_results['Transaction Volumes'].sum()
        average_last_6months = detailed_results['Full Work Day'].iloc[-6:].mean()
        average_upto_last_6months = detailed_results['Full Work Day'].iloc[:-6].mean()
        
        # generate summary output table
    
        #filtered_data.head()
        
        list_record = {'RU Milestone':filtered_data.iloc[0,6],
                       'RU':filtered_data.iloc[0,1],
                       'RU Description':filtered_data.iloc[0,2],
                       'Milestone':filtered_data.iloc[0,3], 
                       'CP Amount':current_period_amount,
                       'CP Predicted':current_period_predicted,
                       'Variance Upto Last 6 months':MAE_upto_6months, 
                       'Variance Last 6 months':MAE_last_6months, 
                       'CP Z-Score':z_score ,
                       'Gradient Upto Last 6 months':gradient_upto_last_6_months[0],
                       'Intercept Upto Last 6 months':intercept_upto_last_6_months,
                       'Gradient Last 6 months':gradient_last_6_months[0],
                       'Intercept Last 6 months':intercept_last_6_months,
                       'Transaction Volumes':transaction_volumes,
                       'r2 Last 6 Months':r_squared_last_6months,
                       'r2 Upto Last 6 Months':r_squared_upto_6months,
                       'Average last 6 months':average_last_6months,
                       'Average upto the last 6 months':average_upto_last_6months
                       }
        
        output_table = output_table.append(list_record, ignore_index=True) 

    
# Output individual model predictions and input data to file for analysis in Tableau

output_table.to_csv("Summary Output.csv")   
output_table_detailed.to_csv("Detailed Output.csv")
print('finished')

