import AFM_SINDy_algorithm_training as phd

# Importing Required Packages
import numpy as np
import math
import pandas as pd
import dill
import cvxpy
import pysindy as ps

# SciPy & NumPy
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# Sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import (
    LinearRegression, Lasso, ElasticNet, Ridge, 
    OrthogonalMatchingPursuit, Lars, LassoLars)

# Matplotlib & Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image

# Miscellaneous
import os
from tqdm import tqdm
from datetime import datetime
import itertools
import multiprocessing
import time
from joblib import Parallel, delayed

# PySINDy utilities
from pysindy.utils import lorenz, lorenz_control, enzyme

### -------------------- 0. AFM Model Functions --------------------

# AFM viscoelastic damp function
def AFM_w_DMT_viscoelastic_damp(t, y, a0, d1, d2, C1, C2, C3, B1, y_bar, forcing_freq):
    
    eta_1, eta_2, phase = y

    deta1dt = eta_2

    if 1-eta_1 <= a0: 
        deta2dt = (-d2*eta_2) - eta_1 - C1 - (C1/a0**2) - C2* (a0-(1 - eta_1))**1.5 - C3*(((a0-(1 - eta_1))**(0.5))*eta_2) + y_bar * (forcing_freq) ** 2 * B1 * np.sin(phase) #Repulsive regime
    else: 
        deta2dt = (-d1*eta_2) - eta_1 - C1 - (C1/(1-eta_1)**2) + y_bar * (forcing_freq) ** 2 * B1 * np.sin(phase)  #Attractive regime

    dphase_dt = forcing_freq

    return [deta1dt, deta2dt,dphase_dt] 

def run_sindy_model(args):
    try:
        (i, combination_lambda_nu, cluster_num, mult_traj_clusters_sections, lib_concat, feature_names,
        constraint_rhs_att, constraint_lhs_att, dt, traject, counter, lock, number_lambda_nu_comb, script_dir) = args

        lambda_value, nu_value = combination_lambda_nu
        serial_number = phd.generate_serial_number(i, cluster_num, lambda_value, nu_value)

        print(f'Currently training cluster {cluster_num} with lambda: {lambda_value} and nu: {nu_value}')

        phd.fit_constrained_SINDy_model(
            candidate_func_library=lib_concat,
            model_feature_names=feature_names,
            constraint_rhs_array=constraint_rhs_att,
            constraint_lhs_array=constraint_lhs_att,
            trajectories_data=mult_traj_clusters_sections[traject][cluster_num],
            traject_dt=dt,
            lambda_val=lambda_value,
            nu_val=nu_value,
            ensemble=True,
            cluster_num=cluster_num,
            multiple_trajectories=True,
            script_dir = script_dir,
            save_folder_name=f'attractive_lib_reduc_viscDMT', #/cluster_{cluster_num}' #for additional folder per cluster
            model_filename=f"constr_attmodel_DMTvisc_SN_{serial_number}",
            save_model=True  
        )
        print(f'Finished training cluster {cluster_num}')
        print(' ')

        with lock:
            counter.value += 1
            print(f"Progress: {counter.value}/" + str(number_lambda_nu_comb * len(cluster_nums)) + " models completed", flush=True)

    except Exception as e:
        print(f"Error in process {i}, cluster {cluster_num}: {e}")

script_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    # Define number of available CPU cores
    num_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())) - 1
    print('The number of used cores is: ' + str(num_cores))

    start_time = time.time()    

    ### -------------------- 1. Data Generation Functions --------------------

    C1 = -1.27462*10**(-6)
    C2 =  4.63118
    C3 = -4e-1 
    B1 = 1.56598
    a0 = 0.0132626 
    d1 = 0.0034 / 2
    d2 = 4.057 / 2
    forcing_freq = 1.002 
    y_bar = 0.05585 
    F_act = B1*(forcing_freq**2)*y_bar 

    dt = 0.01 
    t_train_DMT = np.arange(0, 1000, dt) #time until 1000 steps
    init_cond_DMT = [0.0, 0.0, 0.0] # Initial conditions for position and velocity respectibly
    t_train_span_DMT = (t_train_DMT[0], t_train_DMT[-1])  # Time span for the simulation

    AFM_DMT_ivp = solve_ivp(
        AFM_w_DMT_viscoelastic_damp,
        t_train_span_DMT,
        init_cond_DMT,
        t_eval=t_train_DMT,
        args=(a0, d1, d2, C1, C2, C3, B1, y_bar, forcing_freq),
        rtol=1e-6,
        atol=1e-8,
        method='RK45',
        dense_output=True
    )

    x_train_DMT = AFM_DMT_ivp.y.T
    t_train_DMT_ivp = AFM_DMT_ivp.t

    ### ------ Cantilever beam data  ------
    
    eta_star = 8.88249 #in [nm] for a system: Si-Si Region II (Rützel, S. & Raman, A., 2002)
    w_0_cant = 2*np.pi*11.086e3 #first natural frequency of cantilever in [rad/s]
    Q_factor = 100 #In air for a system: Si-Si Region II (Rützel, S. & Raman, A., 2002)

    ### ------ Generate multiple initial condition trajectories   ------

    x_train_DMT_mult_traj, init_cond_list = phd.generate_different_trajectories(
        dynamical_system=AFM_w_DMT_viscoelastic_damp,
        t_train=t_train_DMT,
        DOF=3,
        num_trajectories=3,
        noisy_trajectories=False,
        parameters=(a0, d1, d2, C1, C2, C3, B1, y_bar, forcing_freq),
        noise_level=None,
        init_cond_range=[(-0.5, 0.5), (-0.25, 0.25), (0, 2 * np.pi)])
    
    traject = 0 #trajectory for training

    print('All multiple trajectories have been successfully created')

    ### -------------------- 2. Cluster Generation Functions --------------------

    mult_traj_cluster_centers_x, mult_traj_cluster_centers_y = phd.define_cluster_centers(x_train_mult_trajectories = x_train_DMT_mult_traj, num_points_high_res=800, 
                                                              sub_sample_val=30, filter_vel_max_val = 0.7, index_to_plot = 0, centers_loc = 3,
                                                              plot_file_name = None, plot=True, save_plot=False, multiple_init_cond =True)

    # Flatten the lists and combine each corresponding x and y into a single list of tuples
    combined_x = np.hstack(mult_traj_cluster_centers_x)
    combined_y = np.hstack(mult_traj_cluster_centers_y)

    # Filter points
    filtered_x, filtered_y = phd.filter_close_points(combined_x, combined_y)

    mult_traj_clusters_sections, mult_traj_clusters_dots_list = phd.generate_fixed_clusters_from_centers(
        cluster_size= 17, x_train_mult_trajectories = x_train_DMT_mult_traj[0],
        KNN_neighbors_num = 400, KNN_radius = 1, 
        mult_traj_cluster_centers_x = filtered_x, 
        mult_traj_cluster_centers_y = filtered_y,  
        multiple_init_cond = False)
    

    cluster_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    
    ### -------------------- 3. Preliminar SINDy Analysis --------------------

    ### ------ Creation of library of candidate functions  ------

    library_functions, library_function_names = phd.create_candidate_func_w_func_names(DOF = 3, poly= True, poly_degrees=[1], 
                                                                                    sin = True, cos = False, n_frequencies=[1], 
                                                                                    AFM_amp=F_act, special = False, special_degrees= None,
                                                                                    AFM_LJ_z=False, poly_frac=False,
                                                                                    exp_damp = False, z_b_val = 0.15, exp_damp_degrees = [1, 2],
                                                                                    AFM_z = False, AFM_z_degrees = [2],
                                                                                    AFM_z_damp=True, AFM_z_damp_degrees_e1 = [1, 2], AFM_z_damp_degrees_e2 = [1, 2],
                                                                                    a0_val = a0, DMT_rep = True, DMT_rep_smooth = False, DMT_rep_degrees = [1.5, 2, 2.5],
                                                                                    DMT_att = True, DMT_att_degrees = [1,2,3,4,5,6,7,8],
                                                                                    DMT_viscoel_damp=True, DMT_visc_elast_degrees_e1 = [0.5, 1], DMT_visc_elast_degrees_e2 = [1, 2],
                                                                                    sin_denominator=False, cos_denominator=False)


    lib_concat = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names, interaction_only=True, include_bias=True)
    lib_concat.fit(x_train_DMT_mult_traj[0])

    feature_names = ['e1', 'e2', 'phase']
    model_DMT_normal = ps.SINDy(feature_names=feature_names, feature_library=lib_concat)
    model_DMT_normal.fit(mult_traj_clusters_sections[traject][4], t=dt, ensemble=False, multiple_trajectories = True, quiet = True) 

    ### ------ Creation of library of candidate functions in Pandas Dataframe ------

    SINDy_cand_func_df= pd.DataFrame(columns=["Candidate Function"])

    for idx, cand_func in enumerate(model_DMT_normal.get_feature_names()):
        SINDy_cand_func_df.loc[idx] = [cand_func]

    print('Candidate function library successfully created')

    ### -------------------- 4. SINDy Regime Analysis assuming Attractive Regime --------------------

    ### ------ Constrain construction for Contrained SINDy Analysis ------

    SINDy_constraints_df = pd.DataFrame(columns=["Equation No.", "Candidate Function", "Constraint Value"])

    #Special constraints for e1' equation:
    SINDy_constraints_df.loc[0] = [0, 'e2', 1]

    #Special constraints for e2' equation:
    SINDy_constraints_df.loc[1] = [1, 'e1', -1]
    SINDy_constraints_df.loc[2] = [1, 'e2', -d1]
    SINDy_constraints_df.loc[3] = [1, 'phase', 0]
    SINDy_constraints_df.loc[4] = [1, 'sin(1*phase)', F_act]
    SINDy_constraints_df.loc[5] = [1, 'sin(1*e1)', 0]
    SINDy_constraints_df.loc[6] = [1, 'sin(1*e2)', 0]
    SINDy_constraints_df.loc[7] = [1, '(1-e2)^-1', 0]
    SINDy_constraints_df.loc[8] = [1, '(1-e2)^-2', 0]
    SINDy_constraints_df.loc[9] = [1, '(1-e2)^-3', 0]
    SINDy_constraints_df.loc[10] = [1, '(1-e2)^-4', 0]
    SINDy_constraints_df.loc[11] = [1, '(1-e2)^-5', 0]
    SINDy_constraints_df.loc[12] = [1, '(1-e2)^-6', 0]
    SINDy_constraints_df.loc[13] = [1, '(1-e2)^-7', 0]
    SINDy_constraints_df.loc[14] = [1, '(1-e2)^-8', 0]

    #Special constraints for phase' equation:
    SINDy_constraints_df.loc[15] = [2, '1', 1]

    #The rest of the constraints are done here:
    cand_func_to_const_in_e1_dot = SINDy_cand_func_df.loc[
                                (SINDy_cand_func_df["Candidate Function"] != 'e2'), "Candidate Function"
                                ].tolist() #Creates a list of candidate functions excluding the function 'e2'. 

    cand_func_to_const_in_phase_dot = SINDy_cand_func_df.loc[
                                (SINDy_cand_func_df["Candidate Function"] != '1'), "Candidate Function"
                                ].tolist() #Creates a list of candidate functions excluding the function '1'. 

    #Normal Constraints for equation e1':
    i = SINDy_constraints_df.shape[0]
    for cand_func_2_const_e1_dot in cand_func_to_const_in_e1_dot:
        SINDy_constraints_df.loc[i] = [0, cand_func_2_const_e1_dot, 0] # In this line, the constraints are placed as: [equation No. 0, cand_func, constraint_value]
        i+=1
        
    #Normal Constraints for equation phase':
    j = SINDy_constraints_df.shape[0]
    for cand_func_2_const_phase_dot in cand_func_to_const_in_phase_dot:
        SINDy_constraints_df.loc[j] = [2, cand_func_2_const_phase_dot, 0] # In this line, the constraints are placed as: [equation No. 2, cand_func, constraint_value]
        j+=1

    lib_concat = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names, interaction_only=True, include_bias=True)
    lib_concat.fit(x_train_DMT_mult_traj[0])

    #Set generalized coordinates for the constrained problem.
    feature_names = ['e1', 'e2', 'phase']
    n_targets = len(feature_names)
    n_features = len(lib_concat.get_feature_names())

    constraint_rhs_att = SINDy_constraints_df["Constraint Value"].to_numpy()
    constraint_lhs_att = np.zeros((len(constraint_rhs_att), n_targets * n_features)) # One row per constraint, one column per coefficient

    #Here the costraints are transformed from a Pandas dataframe to numpy arrays. 
    for index_number in SINDy_constraints_df.index.to_list():
        cand_funct_index = phd.get_index_of_cand_function(model_DMT_normal, SINDy_constraints_df.iloc[index_number].tolist()[1])
        pos_in_matrix = SINDy_constraints_df.iloc[index_number].tolist()[0] * n_features
        constraint_lhs_att[index_number, cand_funct_index + pos_in_matrix ]= 1 
    
    # Open a new .txt file in write mode
    with open('constraints_output.txt', 'w') as file:
        # Loop through the DataFrame and write print output to the file
        for idx, cand_func in enumerate(SINDy_constraints_df.loc[SINDy_constraints_df["Equation No."] == 1, "Candidate Function"].to_list()):
            contraint_value_print = SINDy_constraints_df.loc[
                (SINDy_constraints_df["Equation No."] == 1) & 
                (SINDy_constraints_df["Candidate Function"] == cand_func), 
                "Constraint Value"
            ].iloc[0]
            
            # Redirect print to the file
            print('Constraint ' + str(idx+1) + ': ' + cand_func + ' -> ' + str(round(contraint_value_print, 4)) + " for equation e2'", file=file)

    print('Contraints have been successfully created ')
    ### -------------------- 5. Training multiple models in multiple clusters --------------------

    ### ------ Creating multiple combinations of lambda and nu  reg parameters ------

    threshold_values = phd.generate_random_lambda_n_nu(num_values=7, 
                                                   smaller_order_magnitude=-7, 
                                                   bigger_order_magnitude=-1, randomize=True)
    nu_values = phd.generate_random_lambda_n_nu(num_values=7, 
                                               smaller_order_magnitude=-3, 
                                               bigger_order_magnitude=-1, randomize=True)

    lambda_nu_combinations = list(itertools.product(threshold_values, nu_values))
    number_lambda_nu_comb = len(lambda_nu_combinations)

    ### ------ Creating counter and lock for progress tracking ------

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)  # Shared integer value
    lock = manager.Lock()

    print('All calculations setup have been successfully created')

    ### ------ Prepare arguments for each cluster and each lambda/nu combination ------

    args = []
    for cluster_num in cluster_nums:
        for i, combination in enumerate(lambda_nu_combinations):
            args.append((i, combination, cluster_num, mult_traj_clusters_sections, lib_concat, feature_names,
                        constraint_rhs_att, constraint_lhs_att, dt, traject, counter, lock, number_lambda_nu_comb, script_dir))

    
    ### ------ Creating Paralelization option with joblib ------

    Parallel(n_jobs=num_cores)(
        delayed(run_sindy_model)(arg) for arg in args)

    ### ------ Checking how much time did it take to train all clusters with all lambda and nu combinations ------

    end_time = time.time()
    total_time = end_time - start_time

    # Convert to minutes and seconds for readability
    minutes, seconds = divmod(total_time, 60)
    print(f"Total execution time: {int(minutes)} minutes and {int(seconds)} seconds")
    print('')
    print("SINDy model fitting completed using parallel processing.")