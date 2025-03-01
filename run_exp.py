import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from botorch.test_functions import Beale,Branin,SixHumpCamel,Hartmann,DixonPrice,Rosenbrock,Ackley,Powell,StyblinskiTang
from botorch.utils.transforms import normalize

from OBCGP.utilis import obj_fun, get_initial_points
from OBCGP.GP import ELBO,acq_fcn_EI,par_update
import obj_functions.push_problems
from obj_functions.obj_function import PDEVar
import torch


# create the repo for experiment results
repo_name = "result"

if not os.path.exists(repo_name):
    os.makedirs(repo_name)
    print(f"Directory '{repo_name}' created.")
else:
    print(f"Directory '{repo_name}' already exists.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


function_information = []


temp={}
temp['name']='Branin2D' 
temp['function'] = Branin(negate=True)
temp['fstar'] =  -0.397887 
function_information.append(temp)

# temp={}
# temp['name']='Beale2D' 
# temp['function'] = Beale(negate=True)
# temp['fstar'] =  0.
# function_information.append(temp)

# temp={}
# temp['name']='SixHumpCamel2D' 
# temp['function'] = SixHumpCamel(negate=True)
# temp['fstar'] =  1.0317
# function_information.append(temp)

# temp={}
# temp['name']='Hartmann3D' 
# temp['function'] = Hartmann(dim=3,negate=True)
# temp['fstar'] =  3.86278
# function_information.append(temp)


# temp={}
# temp['name']='Rosenbrock4D' 
# fun_Rosenbrock4D = Rosenbrock(dim=4,negate=True)
# fun_Rosenbrock4D._bounds = [(-2.048, 2.048) for _ in range(4)] 
# fun_Rosenbrock4D.bounds = torch.tensor([[-2.048,-2.048, -2.048, -2.048],
#         [2.048, 2.048, 2.048, 2.048]])

# temp['function'] =  fun_Rosenbrock4D
# temp['fstar'] =  0. 
# function_information.append(temp)

# temp={}
# temp['name']='Ackley6D' 
# temp['function'] = Ackley(dim=6,negate=True)
# temp['fstar'] =  0. 
# function_information.append(temp)


# temp={}
# temp['name']='Powell8D' 
# temp['function'] = Powell(dim=8,negate=True)
# temp['fstar'] = 0. 
# function_information.append(temp)

# temp={}
# temp['name']='StyblinskiTang10D' 
# temp['function'] = StyblinskiTang(dim=10,negate=True)
# temp['fstar'] =  10*39.166166  
# function_information.append(temp)


# temp={}
# temp['name']='PDEVar' 
# temp['function'] = PDEVar()
# temp['fstar'] =  0.
# function_information.append(temp)

# temp={}
# temp['name']='Push4D' 
# f_class = obj_functions.push_problems.push4
# tx_1 = 3.5; ty_1 = 4
# fun = f_class(tx_1, ty_1)
# temp['function'] = fun
# temp['fstar'] =  0.
# function_information.append(temp)



for information in function_information:

    fun = information['function']
    fstar = information['fstar']
   
    neg_fun = obj_fun(fun)

    dim = neg_fun.dim
    bounds = neg_fun.bounds
    print('bounds ',bounds)
    n_init = 4*dim
    priori_known_mode = True
    eps = 1e-5
    Max_iter = 20
    lr = 1e-1


    upp_bdd_prior = fstar


    if dim <=5:
        step_size = 3
        iter_num = 100
        N = 100
    elif dim<=8:
        step_size = 3
        iter_num = 150
        N = 100
    else:
        step_size = 3
        iter_num = 200
        N = 100


    OBCGP = []

    for exp in range(N):

        
        print('exp id: ',exp)

        X = get_initial_points(bounds=bounds,num=n_init,device=device,dtype=dtype,seed=exp)
        X = normalize(X, bounds)
        X = X.numpy()
        Y = neg_fun(X)


        best_record = []
        best_record.append(-np.max(Y))
        print('initial best is: ',best_record[-1])

        for iter in range(iter_num):

            tf.set_random_seed(1234)
            np.random.seed(1234)
            
            if iter%step_size == 0:
                my_std = np.std(Y)
                my_mean = np.mean(Y)
            
 
            Y_rescale = (Y-my_mean)/my_std
            upp_bdd_prior_rescale = (upp_bdd_prior-my_mean)/my_std 

            max_rescale = np.max(Y_rescale)
            low_bdd_feed = max_rescale
            upp_bdd_feed = upp_bdd_prior_rescale-max_rescale

            lamda_feed = np.float32(np.log(10.0))
            alpha_feed = np.float32(0.0)
            beta_feed = lamda_feed
            X_M_t = np.reshape(X[np.argmax(Y)], (1, dim))
            X_M_t_feed = np.log(np.divide(X_M_t, 1.0 - X_M_t + eps)) + np.random.normal(0.0, 0.01, [1, dim])
        
        
            
            if iter%step_size == 0: # this is a warm-up for model training by VI
                try:
                    sigma_feed, phi_feed = par_update(X, Y_rescale,dim)
                except:
                    print('use old hyperparameters')
                    pass
            
            # we are transforming some parameters that needs training to be Variables
            tf.reset_default_graph()
            alpha_v = tf.Variable(tf.convert_to_tensor(alpha_feed, dtype=tf.float32))
            beta_v = tf.Variable(tf.convert_to_tensor(beta_feed, dtype=tf.float32))
            X_M_t_v = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
            X_M = (tf.sigmoid(X_M_t_v))
            x_data_p = tf.convert_to_tensor(X, dtype=tf.float32)
            y_data_p = tf.convert_to_tensor(Y_rescale, dtype=tf.float32)
            low_bdd_p = tf.convert_to_tensor(low_bdd_feed, tf.float32)
            upp_bdd_p=tf.convert_to_tensor(upp_bdd_feed,tf.float32)
            lamda_p = tf.convert_to_tensor(lamda_feed, dtype=tf.float32)
            phi_p = (tf.convert_to_tensor(phi_feed, dtype=tf.float32))
            sigma_p = (tf.convert_to_tensor(sigma_feed, dtype=tf.float32))

            pt_sel = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
            costF = ELBO(x_data_p, y_data_p, low_bdd_p,upp_bdd_p, X_M, phi_p, sigma_p, alpha_v, beta_v, lamda_p)
            q_mu_p = tf.exp(alpha_v) / (tf.exp(alpha_v) + tf.exp(beta_v))
            q_var_p = tf.divide(tf.exp(alpha_v+beta_v),tf.square(tf.exp(alpha_v)+tf.exp(beta_v))*(1.0+tf.exp(alpha_v)+tf.exp(beta_v)))
            acq_fcn_val = acq_fcn_EI(tf.sigmoid(pt_sel), x_data_p, y_data_p, low_bdd_p,upp_bdd_p, X_M, phi_p, sigma_p,q_mu_p,q_var_p)

            try:
                optimizer = tf.train.AdamOptimizer(lr)  #tf.train.GradientDescentOptimizer(lr)
            
                train_par = optimizer.minimize(costF)
                train_acq = optimizer.minimize(-acq_fcn_val, var_list=pt_sel)

                sess = tf.Session()
                init = tf.global_variables_initializer()
                sess.run(init)

                for opt_iter in range(Max_iter):
                    sess.run(train_par)
                for opt_iter_acq in range(Max_iter):
                    sess.run(train_acq)
                X_M_np, phi_np, sigma_np, alpha_np, beta_np = sess.run((X_M, phi_p, sigma_p, alpha_v, beta_v))
                q_mu = np.exp(alpha_np - beta_np)
                q_var = np.exp(alpha_np - 2.0 * beta_np)
                pt_next = sess.run(tf.sigmoid(pt_sel))
                val_next = np.reshape(neg_fun(pt_next), [])
                
            except:
 
                print('error in opt! we do random search!')
                X_random = get_initial_points(bounds=bounds,num=1,device=device,dtype=dtype,seed=iter)
                X_random = normalize(X_random, bounds)
                pt_next = X_random.numpy()
                val_next = np.reshape(neg_fun(pt_next), [])
            
                
            X = np.concatenate((X, pt_next), axis=0)
            Y = np.concatenate((Y, np.reshape(val_next, [1, 1])), axis=0)

            sess.close()
            tf.reset_default_graph()
            X_M_best = np.reshape(X[np.argmax(Y)], (1, dim))

            best_record.append( -np.max(Y))
            
            print('iteration #:%d\t' % (iter + 1) + 'current Minimum value: %f\t' % (-np.max(Y)) )

        OBCGP.append(best_record)
        
        np.savetxt('result/'+information['name']+'_OBCGP', OBCGP, delimiter=',')
    