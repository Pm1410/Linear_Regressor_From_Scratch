import numpy as np
class Error_Metrics:

    @staticmethod   
    def mse(y,y_pred):
        return np.mean((y-y_pred)**2)

    @staticmethod   
    def rmse(y,y_pred):
        return np.sqrt(np.mean((y-y_pred)**2))
    
    @staticmethod   
    def mae(y,y_pred):
        return np.mean(np.abs(y - y_pred))
    
    @staticmethod   
    def r2_score(y,y_pred):
        y_mean=np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        return 1 - ss_res / ss_tot