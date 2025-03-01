import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from numpy import genfromtxt
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.preprocessing import StandardScaler


    
class XGBoost:
    def __init__(self,task,seed=1):
        # define the search range for each variable
        self.bounds = torch.tensor(np.asarray([
                                [0.,10.],  # alpha
                                  [0.,10.],# gamma 
                                  [5.,15.], #max_depth
                                  [1.,20.],  #min_child_weight
                                  [0.5,1.],  #subsample
                                  [0.1,1] #colsample
                                 ]).T)
            
        self.dim = 6
        self.fstar = 100

        self.seed= seed
        
        self.task = task
        
        if task == 'skin':
            data = np.genfromtxt('obj_functions/Skin_NonSkin.txt', dtype=np.int32)
            outputs = data[:,3]
            inputs = data[:,0:3]
            X_train1, _, y_train1, _ = train_test_split(inputs, outputs, test_size=0.85, random_state=self.seed)
            y_train1 = y_train1-1
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
            
        elif task == 'bank':
            data = pd.read_csv('obj_functions/BankNote_Authentication.csv')
            X = data.loc[:, data.columns!='class']
            y = data['class']
            outputs = np.array(y)
            inputs = np.array(X)
            X_train1, _, y_train1, _ = train_test_split(inputs, outputs, test_size=0.05, random_state=self.seed)
            
            self.X_train1 = X_train1
            self.y_train1 = y_train1
 
        elif task == 'breast':
            df_data = pd.read_csv('obj_functions/Breast_Cancer_Wisconsin.csv')
            class_mapping = {'M':0, 'B':1}
            df_data['diagnosis'] = df_data['diagnosis'].map(class_mapping)
            
            df_data=df_data.drop(['id','Unnamed: 32'],axis=1)
            
            df_data = df_data.drop([  'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                'fractal_dimension_se'],axis=1)
            
            X = df_data.loc[:, df_data.columns!='diagnosis']
            y = df_data['diagnosis']


            outputs = np.array(y)
            inputs = np.array(X)
      
            X_train1, _, y_train1, _ = train_test_split(inputs, outputs, test_size=0.05, random_state=self.seed)

            self.X_train1 = X_train1  
            self.y_train1 = y_train1

            
    def __call__(self, X):
        
        X = X.numpy().reshape(6,)

        
        alpha,gamma,max_depth,min_child_weight,subsample,colsample=X[0],X[1],X[2],X[3],X[4],X[5]
        
        if self.task == 'bank'  or self.task == 'skin' or self.task == 'breast':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                        min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=self.seed, objective = 'binary:logistic', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
        elif self.task == 'iris':
            reg = XGBClassifier(reg_alpha=alpha, gamma=gamma, max_depth=int(max_depth), subsample=subsample, 
                    min_child_weight=min_child_weight,colsample_bytree=colsample, n_estimators = 2, random_state=1, objective = 'multi:softmax', booster='gbtree',eval_metric='logloss',silent=None)
            score = np.array(cross_val_score(reg, X=self.X_train1, y=self.y_train1).mean())
      
        return - (100-torch.tensor([score*100]))
    


################################ PDE test function #########################################################

import psutil
import os
import torch
from torch import Tensor
import gc
import tempfile
from multiprocessing import Process, Event
try:
    from pde import PDE, FieldCollection, ScalarField, UnitGrid
except ImportError:
    pass


class PDEVar:
    def __init__(self,):
        self.dim = 4
        self.bounds  = torch.tensor([
            [0.1, 5.0],
            [0.1, 5.0],
            [0.01, 5.0],
            [0.01, 5.0],
        ]).T


    @staticmethod
    def simulator_process(tensor, temp_file, done_event):
        try:
            torch.manual_seed(1234)
            a = tensor[0].item()
            b = tensor[1].item()
            d0 = tensor[2].item()
            d1 = tensor[3].item()

            eq = PDE(
                {
                    "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                    "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
                }
            )

            grid = UnitGrid([64, 64])  # Simulation grid
            u = ScalarField(grid, a, label="Field $u$")
            v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
            state = FieldCollection([u, v])

            # Solve the PDE
            sol = eq.solve(state, t_range=20, dt=1e-3)

            sol_tensor = torch.stack(
                (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
            )

            # Replace invalid values with random noise
            sol_tensor[~torch.isfinite(sol_tensor)] = 1e4 * torch.randn_like(
                sol_tensor[~torch.isfinite(sol_tensor)]
            )

            # Save the result
            torch.save({"success": True, "result": sol_tensor}, temp_file)
        except Exception as e:
            # Handle any errors
            torch.save({"success": False, "error": str(e)}, temp_file)
        finally:
            # Signal that the process is done
            done_event.set()
            del sol, u, v, state, eq, grid
            gc.collect()


    def Simulator(self, tensor, timeout=90):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name

        # Event to signal completion
        done_event = Event()

        # Run the simulation in a separate process
        process = Process(
            target=self.simulator_process, args=(tensor, temp_file_name, done_event)
        )
        process.start()

        # Wait for the process to complete or timeout
        process.join(timeout=timeout)

        # Check if the process is still alive (timed out)
        if process.is_alive():
            process.terminate()  # Forcefully terminate the process
            process.join()  # Ensure proper cleanup
            print(
                f"Simulation timed out after {timeout} seconds. Returning random fallback value."
            )
            # Return a random tensor as a fallback
            return torch.randn(1,2, 64, 64)

        # Ensure the process signals completion
        if not done_event.is_set():
            print(
                "Simulation process did not complete as expected. Returning random fallback value."
            )
            return torch.randn(1,2, 64, 64)

        # Load the result
        result = torch.load(temp_file_name)
        os.remove(temp_file_name)  # Clean up the temporary file

        if result["success"]:
            return result["result"]
        else:
            print(f"Simulation failed with error: {result['error']}. Returning random fallback value.")
            # Return a random tensor as a fallback
            return torch.randn(1,2, 64, 64)


    def __call__(self, X):
        with torch.no_grad():
            sims = torch.stack([self.Simulator(x) for x in X]).to(X.device)

        sz = sims.shape[-1]
        weighting = (
            torch.ones(2, sz, sz, device=sims.device, dtype=sims.dtype) / 10
        )
        weighting[:, [0, 1, -2, -1], :] = 1.0
        weighting[:, :, [0, 1, -2, -1]] = 1.0

        sims = sims * weighting
        res = 100 * sims.var(dim=(-1, -2, -3)).detach()
        

        print(sims)
        # Clean up memory
        del sims
        gc.collect()


        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.2f} MB")


        return -res



