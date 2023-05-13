from torch import sqrt, abs


def getMSE(y_hat, y):
    mse = ((y - y_hat)*(y - y_hat)).mean()
    return mse.detach().cpu().numpy()
    
def getMAE(y_hat, y):
    mae = abs(y_hat - y).mean()
    return mae.detach().cpu().numpy()