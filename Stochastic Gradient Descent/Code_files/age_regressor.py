import numpy as np

def train_age_regressor (mini_batch_size,learning_rate,num_epochs):
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    #X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    #yte = np.load("age_regression_yte.npy")
    
    num_datapoints = X_tr.shape[0] 
    
    split_index = int(0.8 * num_datapoints)
    
    indices = np.arange(num_datapoints)
    np.random.shuffle(indices)
    
    train_indices = indices[:split_index]
    val_indices = indices[split_index:] 
    
    X_train, X_val = X_tr[train_indices], X_tr[val_indices]   
    y_train, y_val = ytr[train_indices], ytr[val_indices]
    
    X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
    y_train = (y_train - np.mean(y_train, axis=0)) / (np.std(y_train, axis=0) + 1e-8)
    
    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    
    num_tr_samples = X_train.shape[0]
    tr_indices = np.arange(num_tr_samples)
    
    mse_list = []
    
    for epoch in range(num_epochs):
        np.random.shuffle(tr_indices)
        
        for num in range(0,num_tr_samples,mini_batch_size):
            batch_index = tr_indices[num: num + mini_batch_size]
            
            x_batch = X_train[batch_index]
            y_batch = y_train[batch_index]
            
            error_value = (np.dot(x_batch,weights) + bias) - y_batch 
            
            error_value = np.nan_to_num(error_value, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
            
            mse = np.mean(error_value ** 2)
            
            if np.isinf(mse) or np.isnan(mse):
                print(f"Invalid MSE detected at Epoch {epoch+1}, Batch {num//mini_batch_size + 1}. Skipping batch...")
                continue
            
            mse_list.append(mse)
            #print(f"Epoch {epoch+1}, Batch {num//mini_batch_size + 1}, MSE: {mse}")
            
            weight_grad = np.dot(x_batch.T,(np.dot(x_batch,weights) + bias) - y_batch )/x_batch.shape[0]
            
            bias_grad = np.mean(error_value)
            
            
            weights -= learning_rate * weight_grad 
            bias -= learning_rate * bias_grad
        
        
    return weights,bias,X_val,y_val


def validation():
    
    learning_rates = [1e-5,1e-4,1e-3] 
    mini_batch_sizes = [32, 64,128]
    num_epochs_testing = [50, 100,150]
    
    best_mse = float('inf')
    best_hyperparams = {}
    best_weights, best_bias = None, None
    
    
    
    for rate in learning_rates:
        for batch in mini_batch_sizes:
            for epoch in num_epochs_testing:
                
                weights, bias, X_val, y_val = train_age_regressor(mini_batch_size=batch, learning_rate=rate,num_epochs=epoch)
                
                X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0) 
                y_val = (y_val - np.mean(y_val, axis=0)) / np.std(y_val, axis=0) 
                
                # once we have the trained weights we validate the model
                
                y_val_pred = np.dot(X_val, weights) + bias
                mse = np.mean((y_val_pred - y_val) ** 2)  # mean squared error to validate prediction
                #print(mse)
                #print(f"Num_Epoch {epoch}, Batch_size {batch}, Learning_rate {rate}, MSE: {mse}")
                if mse < best_mse:  
                    best_mse = mse
                    best_hyperparameters = {'num_epochs': epoch,'learning_rate': rate,'mini_batch': batch}
                    best_weights,best_bias = weights,bias
        
        
    return best_hyperparameters,best_weights,best_bias,best_mse  




if __name__ == "__main__":
    
    best_hyp,best_weights,best_bias,best_mse= validation()
    np.save("best_model_weights3.npy", best_weights)
    np.save("best_model_bias3.npy", best_bias)
    np.save("best_model_hyperparameters3",best_hyp)
    
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    X_te = (X_te - np.mean(X_te, axis=0)) / np.std(X_te, axis=0)
    y_te = (y_te - np.mean(y_te, axis=0)) / np.std(y_te, axis=0)
    
    test_weights = np.load("best_model_weights3.npy")
    test_bias = np.load("best_model_bias3.npy")
    
    y_test_pred = np.dot(X_te, test_weights) + test_bias
    
    test_mse = np.mean((y_test_pred - y_te) ** 2)
    print(f"Test MSE: {test_mse}")
  
  
