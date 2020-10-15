#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sophiabj/03-presidential-election/blob/master/Election_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().system('python -m pip install --user --upgrade pip')

get_ipython().system('pip3 install pandas==0.23.4 matplotlib==3.0.3 scipy==1.2.1 scikit-learn==0.22  --user')


# In[9]:


pip install kfp --user


# In[10]:


get_ipython().system('which dsl-compile')


# In[4]:


import kfp
import kfp.dsl as dsl
import kfp.components as comp


# In[5]:


out_dir = "/home/03-presidential-election/"


# In[6]:


def train(data_path, regr_multirf_file):
    
    import pickle
    import sys, subprocess;
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pandas==0.23.4'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn==0.22'])
    import pandas as pd
    import numpy as np
    
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    
    df = pd.read_csv('https://github.com/sophiabj/03-presidential-election/blob/master/data/usa-2016-presidential-election-by-county.csv')

    new_dataframe = df[['Asian','At Least High School Diploma','Black','Child.Poverty.living.in.families.below.the.poverty.line',
                      'Democrats 08 (Votes)','Democrats 12 (Votes)','Democrats 2008','Democrats 2012','Graduate Degree',
                      'Nearest County','Poverty.Rate.below.federal.poverty.threshold','Republicans 08 (Votes)',
                      'Republicans 12 (Votes)','Republicans 2008','Republicans 2012','Total Population','Votes',
                      'White','White  Asian','total08','total12','total16', 'Democrats 2016', 'Republicans 2016']]
    
    
    
    X = new_dataframe.drop(columns=['Democrats 2016', 'Republicans 2016'])
    y = new_dataframe[['Democrats 2016', 'Republicans 2016']]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    scaler = MinMaxScaler()

    normalised_train_df = scaler.fit_transform(x_train)
    normalised_train_df = pd.DataFrame(normalised_train_df, columns=x_train.columns)

    normalised_test_df = scaler.transform(x_test)
    normalised_test_df = pd.DataFrame(normalised_test_df, columns=x_train.columns)


    max_depth = 30
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
    max_depth=max_depth,
    random_state=0))
    regr_multirf.fit(normalised_train_df, y_train)

    regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
    random_state=2)
    regr_rf.fit(normalised_train_df, y_train)
    
    
     #output file to path
    np.savez_compressed(f'{data_path}/preprocessed-data.npz', 
                       xtrain=normalised_train_df,
                       xtest=normalised_test_df,
                       ytrain=y_train,
                       ytest=y_test)
    print("Preprocessing Done")

     #Save the model to the designated 
    with open(f'{data_path}/{regr_multirf_file}', 'wb') as file:
        pickle.dump(regr_multirf, file)
        
    print("Model Trained")


# In[40]:


def predict(data_path, regr_multirf_file):
    
    import pickle
    import os
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    
    with open(f'{data_path}/{regr_multirf_file}','rb') as file:
        regr_multirf = pickle.load(file)

    # Load and unpack the test_data
    preprocessed_data = np.load(f'{data_path}/preprocessed-data.npz')
    x_test = preprocessed_data['xtest']
    y_test = preprocessed_data['ytest']

    #Evaluate the model and print results
    y_multirf = regr_multirf.predict(normalised_test_df)
    y_rf = regr_rf.predict(normalised_test_df)

    r2 = metrics.r2_score(y_test,y_multirf)
    round(r2,2)
    
    print('Model \nr2 score = {}' .format(metrics.r2_score(y_test,y_multirf)))
    
    #np.savetxt
    with open(f'{data_path}/model_result.txt','w') as result:
        result.write(" Prediction: {},\nActual: {} ".format(y_multirf,y_test))
    
    print('Prediction has been saved successfully!')
    


# In[41]:


train_op = comp.func_to_container_op(train , base_image = "tensorflow/tensorflow:latest-gpu-py3")
predict_op = comp.func_to_container_op(predict , base_image = "tensorflow/tensorflow:latest-gpu-py3")


# In[42]:



client = kfp.Client(host='79cc5e1a5388f21b-dot-us-central2.pipelines.googleusercontent.com')


# In[51]:


# Define the pipeline
@dsl.pipeline(
   name='Presidential Elections Pipeline',
   description='An ML pipeline that predicts outcome of presidential elections.'
)

# Define parameters to be fed into pipeline
def presidential_election_pipeline(
    data_path: str,
    regr_multirf_file: str
):
    
    # Define volume to share data between components.
    vop = dsl.VolumeOp(
    name="create_volume",
    resource_name="data-volume", 
    size="1Gi", 
    modes=dsl.VOLUME_MODE_RWO)
    
    # Create presidential elections training component.
    presidential_elections_training_container = train_op(data_path, regr_multirf_file)                                     .add_pvolumes({data_path: vop.volume})

    # Create presidential elections prediction component.
    presidential_elections_predict_container = predict_op(data_path, regr_multirf_file)                                     .add_pvolumes({data_path: presidential_elections_training_container.pvolume})
    
    # Print the result of the prediction
    presidential_elections_result_container = dsl.ContainerOp(
        name="print_prediction",
        image='library/bash:4.4.23',
        pvolumes={data_path: presidential_elections_predict_container.pvolume},
        arguments=['cat', f'{data_path}/result.txt']
    )


# In[52]:


DATA_PATH = '/mnt'
REGR_MULTIFR_PATH='presidential_elections_model.h5'


# In[53]:


pipeline_func = presidential_election_pipeline


# In[54]:


experiment_name = 'preselect'
run_name = pipeline_func.__name__ + ' run'

arguments = {"data_path":DATA_PATH,
             "regr_multirf_file":REGR_MULTIFR_PATH}

# Compile pipeline to generate compressed YAML definition of the pipeline.
kfp.compiler.Compiler().compile(pipeline_func,  
  '{}.zip'.format(experiment_name))

# Submit pipeline directly from pipeline function
run_result = client.create_run_from_pipeline_func(pipeline_func, 
                                                  experiment_name=experiment_name, 
                                                  run_name=run_name, 
                                                  arguments=arguments)


# In[ ]:





# In[ ]:




