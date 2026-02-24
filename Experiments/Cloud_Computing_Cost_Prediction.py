import numpy as np
import pandas as pd

''' This Dataset has 3 Features x1->Compute_cores, x2->data_stored_tb, x3->newtwork_trafiic_gb'''

def generate_cloud_cost_data(n_samples=5000, random_state=42):

    np.random.seed(random_state)

    # genearting the 3 Features of Dataset 

    x1_cores = np.random.uniform(2,64,n_samples) # 2 to 64 cores
    x2_data = np.random.uniform(10,500,n_samples) # 10 to 500 TB
    x3_traffic = np.random.uniform(100,5000,n_samples) # 100 to 5000 GB

    # Defining the true weights

    # lin -> Linear, quad -> Quadratic
    w0_bias = 150.0
    w1_cores_lin = 20.0
    w2_cores_quad = 1.5
    w3_data_lin = 5.0
    w4_traffic_lin = 0.5
    w5_traffic_quad = 0.002

    # Calculating the true target values using the weights defined above. 

    y_true = ( w0_bias + (w1_cores_lin * x1_cores) + (w2_cores_quad * (x1_cores**2)) + (w3_data_lin * x2_data) + (w4_traffic_lin * x3_traffic) + (w5_traffic_quad * (x3_traffic**2)) )

    # Adding Gaussian Noise to simulate the real world data 

    noise_std = 250
    epsilon = np.random.normal(0, noise_std, n_samples)
              
    # Final target values, which are visible to us
    
    t_target = y_true + epsilon

    # Now putting everything of our dataset in a dataframe 

    dataset = pd.DataFrame({
        'Compute_Cores': np.round(x1_cores, 1),
        'Data_Stored_TB': np.round(x2_data, 1),
        'Network_Traffic_GB': np.round(x3_traffic, 1),
        'Monthly_Cloud_Cost': np.round(t_target, 2)
    })

    return dataset


dataset = generate_cloud_cost_data(n_samples=5000)
dataset.to_csv("Cloud_Computing_cost_dataset.csv", index=False)
