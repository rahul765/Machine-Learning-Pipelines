from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import sqlite3

from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Fetching environment variable
dag_config = Variable.get("variables_config", deserialize_json=True)
data_path = dag_config["data_path"]
preprocessed_data = dag_config["preprocessed_data"]
scaled_data = dag_config["scaled_data"]
training_data = dag_config["training_data"]
testing_data = dag_config["testing_data"]
LR_results = dag_config["LR_results"]
DT_results = dag_config["DT_results"]
RF_results = dag_config["RF_results"]
SVR_results = dag_config["SVR_results"]
GBR_results = dag_config["GBR_results"]
BestModel_results = dag_config["BestModel_results"]

# Checking if Data is availabe
def data_is_available(_file_name=data_path, **kwargs):
    dataset = pd.read_csv(_file_name)
    if dataset.empty:
        print("No Data Fetched")
    else:
        print("{} records have been fetched".format(dataset.shape[0]))
    return "{} records have been fetched".format(dataset.shape[0])

# Preprocessing the dataset
def preprocessing(_file_name=data_path, **kwargs):
    dataset = pd.read_csv(_file_name)
    dum_df = pd.get_dummies(dataset, columns=["State"], prefix=["State_is"] )
    dum_df.drop(['State_is_Florida'], axis=1, inplace=True)
    dum_df.to_csv(preprocessed_data)
    print(dum_df.head())

def scaling(_file_name=preprocessed_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y = dataset['Profit']
    X = dataset.drop(['Profit'], axis=1)
    X_colums = X.columns
    # Feature Scaling
    sc_X = MinMaxScaler()
    X = sc_X.fit_transform(X)
    scaled_X = pd.DataFrame(X, columns=X_colums)
    scaled_X['Profit'] = y
    scaled_X.to_csv(scaled_data)
    print(scaled_X.head())

def splitting_data(_file_name=scaled_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y = dataset['Profit']
    X = dataset.drop(['Profit'], axis=1)
    X_colums = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = pd.DataFrame(X_train,columns=X_colums)
    X_train['Profit'] = y_train
    X_test = pd.DataFrame(X_test,columns=X_colums)
    X_test['Profit'] = y_test
    X_train.to_csv(training_data)
    X_test.to_csv(testing_data)
    print("Ratio is 80:20")

def evaluate(y_test,y_pred,test_train):
    MSE_test = mean_squared_error(y_test, y_pred)
    RMSE_test = np.sqrt(MSE_test)
    Adjusted_RSquare_test = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test,y_pred)
    print('Model Performance for {:s}'.format(test_train))
    print('Mean Sqaure Error(MSE): {:0.4f}.'.format(MSE_test))
    print('Root Mean Sqaure Error(RMSE): {:0.4f}.'.format(RMSE_test))
    print('Adjusted R Square = {:0.2f}.'.format(Adjusted_RSquare_test))
    print('MAE = {:0.2f}.'.format(MAE))
    return {"MSE": MSE_test, "RMSE" : RMSE_test, "Adjusted_RSquare_test": Adjusted_RSquare_test, "MAE": MAE}


def model_1_LR(_file_name=training_data, _test_file=testing_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y_train = dataset['Profit']
    X_train = dataset.drop(['Profit'], axis=1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    test = pd.read_csv(_test_file)
    y_test = test['Profit']
    X_test = test.drop(['Profit'], axis=1)
    y_pred = regressor.predict(X_test)
    test_results = evaluate(y_test,y_pred, 'Linear Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(LR_results)
    print(test_results)

def model_2_DT(_file_name=training_data, _test_file=testing_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y_train = dataset['Profit']
    X_train = dataset.drop(['Profit'], axis=1)
    regressor = DecisionTreeRegressor(random_state=123)
    regressor.fit(X_train, y_train)
    test = pd.read_csv(_test_file)
    y_test = test['Profit']
    X_test = test.drop(['Profit'], axis=1)
    y_pred = regressor.predict(X_test)
    test_results = evaluate(y_test,y_pred,'Decision Tree Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(DT_results)
    print(test_results)

def model_3_RF(_file_name=training_data, _test_file=testing_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y_train = dataset['Profit']
    X_train = dataset.drop(['Profit'], axis=1)
    regressor = RandomForestRegressor(random_state=123)
    regressor.fit(X_train, y_train)
    test = pd.read_csv(_test_file)
    y_test = test['Profit']
    X_test = test.drop(['Profit'], axis=1)
    y_pred = regressor.predict(X_test)
    test_results = evaluate(y_test,y_pred,'Random Forest Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(RF_results)
    print(test_results)

def model_4_SVR(_file_name=training_data, _test_file=testing_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y_train = dataset['Profit']
    X_train = dataset.drop(['Profit'], axis=1)
    regressor = SVR()
    regressor.fit(X_train, y_train)
    test = pd.read_csv(_test_file)
    y_test = test['Profit']
    X_test = test.drop(['Profit'], axis=1)
    y_pred = regressor.predict(X_test)
    test_results = evaluate(y_test,y_pred,'Support Vector Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(SVR_results)
    print(test_results)

def model_5_GBR(_file_name=training_data, _test_file=testing_data, **kwargs):
    dataset = pd.read_csv(_file_name)
    y_train = dataset['Profit']
    X_train = dataset.drop(['Profit'], axis=1)
    regressor = GradientBoostingRegressor(random_state=123)
    regressor.fit(X_train, y_train)
    test = pd.read_csv(_test_file)
    y_test = test['Profit']
    X_test = test.drop(['Profit'], axis=1)
    y_pred = regressor.predict(X_test)
    test_results = evaluate(y_test,y_pred,'Gradient Boosting Regression')
    test_results = pd.DataFrame.from_dict(test_results, orient='index',columns=['Test Values'])
    test_results.to_csv(GBR_results)
    print(test_results)

def best_model(m1=LR_results, m2=DT_results, m3=RF_results, m4=SVR_results, m5=GBR_results, best=BestModel_results, **kwargs):
    dt = pd.read_csv(m2)
    rf = pd.read_csv(m3)
    gbr = pd.read_csv(m5)
    lr = pd.read_csv(m1)
    svr = pd.read_csv(m4)
    conn = sqlite3.connect(':memory:')
    dt.to_sql('dt',conn,index=False)
    rf.to_sql('rf',conn,index=False)
    gbr.to_sql('gbr',conn,index=False)
    lr.to_sql('lr',conn,index=False)
    svr.to_sql('svr',conn,index=False)
    tmp=pd.read_sql_query('''select dt."Unnamed: 0" as ErrorMetric,
    case
    when dt."Unnamed: 0"<>'Adjusted_RSquare_test' and dt."Test Values"<rf."Test Values" and dt."Test Values"<gbr."Test Values" and dt."Test Values"<lr."Test Values" and dt."Test Values"<svr."Test Values"
        then 'DT'
    when dt."Unnamed: 0"<>'Adjusted_RSquare_test' and rf."Test Values"<dt."Test Values" and rf."Test Values"<gbr."Test Values" and rf."Test Values"<lr."Test Values" and rf."Test Values"<svr."Test Values"
        then 'RF'
    when dt."Unnamed: 0"<>'Adjusted_RSquare_test' and gbr."Test Values"<dt."Test Values" and gbr."Test Values"<rf."Test Values" and gbr."Test Values"<lr."Test Values" and gbr."Test Values"<svr."Test Values"
        then 'GBR'
    when dt."Unnamed: 0"<>'Adjusted_RSquare_test' and lr."Test Values"<dt."Test Values" and lr."Test Values"<rf."Test Values" and lr."Test Values"<gbr."Test Values" and lr."Test Values"<svr."Test Values"
        then 'LR'
    when dt."Unnamed: 0"<>'Adjusted_RSquare_test' and svr."Test Values"<dt."Test Values" and svr."Test Values"<rf."Test Values" and svr."Test Values"<lr."Test Values" and svr."Test Values"<gbr."Test Values"
        then 'SVR'
    when dt."Unnamed: 0"='Adjusted_RSquare_test' and dt."Test Values">rf."Test Values" and dt."Test Values">gbr."Test Values" and dt."Test Values">lr."Test Values" and dt."Test Values">svr."Test Values"
        then 'DT'
    when dt."Unnamed: 0"='Adjusted_RSquare_test' and rf."Test Values">dt."Test Values" and rf."Test Values">gbr."Test Values" and rf."Test Values">lr."Test Values" and rf."Test Values">svr."Test Values"
        then 'RF'
    when dt."Unnamed: 0"='Adjusted_RSquare_test' and gbr."Test Values">dt."Test Values" and gbr."Test Values">rf."Test Values" and gbr."Test Values">lr."Test Values" and gbr."Test Values">svr."Test Values"
        then 'GBR'
    when dt."Unnamed: 0"='Adjusted_RSquare_test' and lr."Test Values">dt."Test Values" and lr."Test Values">rf."Test Values" and lr."Test Values">gbr."Test Values" and lr."Test Values">svr."Test Values"
        then 'LR'
    when dt."Unnamed: 0"='Adjusted_RSquare_test' and svr."Test Values">dt."Test Values" and svr."Test Values">rf."Test Values" and svr."Test Values">lr."Test Values" and svr."Test Values">gbr."Test Values"
        then 'SVR'
    end as BestModel
    from dt
    inner join rf on dt."Unnamed: 0"=rf."Unnamed: 0"
    inner join gbr on gbr."Unnamed: 0"=rf."Unnamed: 0"
    inner join lr on lr."Unnamed: 0"=gbr."Unnamed: 0"
    inner join svr on svr."Unnamed: 0"=lr."Unnamed: 0"''',conn)
    tmp.to_csv(best)

project_cfg = {
    'owner': 'airflow',
    'email': ['your-email@example.com'],
    'email_on_failure': False,
    'start_date': datetime(2019, 8, 1),
    'retries': 1,
    'retry_delay': timedelta(hours=1),
}

dag = DAG('Regression_ML_Pipeline',
          default_args=project_cfg,
          schedule_interval=timedelta(days=1))

task_1 = PythonOperator(
    task_id='is_data_available',
    provide_context=True,
    python_callable=data_is_available,
#    op_kwargs={'_file_name': '50_Startups.csv'},
    dag=dag,
)

task_2 = PythonOperator(
    task_id='preprocessing',
    provide_context=True,
    python_callable=preprocessing,
    dag=dag,
)

task_3 = PythonOperator(
    task_id='scaling',
    provide_context=True,
    python_callable=scaling,
    dag=dag,
)

task_4 = PythonOperator(
    task_id='splitting_data',
    provide_context=True,
    python_callable=splitting_data,
    dag=dag,
)

task_5_1 = PythonOperator(
    task_id='linear_regression',
    provide_context=True,
    python_callable=model_1_LR,
    dag=dag,
)

task_5_2 = PythonOperator(
    task_id='decision_tree_regression',
    provide_context=True,
    python_callable=model_2_DT,
    dag=dag,
)



task_5_3 = PythonOperator(
    task_id='random_forest_regression',
    provide_context=True,
    python_callable=model_3_RF,
    dag=dag,
)

task_5_4 = PythonOperator(
    task_id='support_vector_regression',
    provide_context=True,
    python_callable=model_4_SVR,
    dag=dag,
)

task_5_5 = PythonOperator(
    task_id='gradient_boosting_regression',
    provide_context=True,
    python_callable=model_5_GBR,
    dag=dag,
)

task_6 = PythonOperator(
    task_id='best_model',
    provide_context=True,
    python_callable=best_model,
    dag=dag,
)

task_1 >> task_2 >> task_3 >> task_4 >> [task_5_1, task_5_2, task_5_3, task_5_4, task_5_5] >> task_6
##
