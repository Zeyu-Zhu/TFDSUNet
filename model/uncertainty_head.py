import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch import log, pi, lgamma


# Negative Log Likelihood
# Prior: Normal Inverse-Gamma (NIG) distribution
def NIG_NLL(y, gamma, nu, alpha, beta):
    omega = 2*beta*(1 + nu)
    nig_nll = 0.5*log(pi/nu) - alpha*log(omega) + (alpha + 0.5)*log((y - gamma)*(y - gamma)*nu + omega) + lgamma(alpha) - lgamma(alpha + 0.5)
    return nig_nll.mean()

# A penalty whenever there is an error in the prediction
# Scales with the total evidence of our inferred posterior
def NIG_Regularization(y, gamma, nu, alpha):
    error = (y - gamma).abs()
    evidence = 2 * nu + alpha
    return (error*evidence).mean()


class UncertaintyHead(nn.Module):
    
    def __init__(self, input_dim):
        super(UncertaintyHead, self).__init__()
        self.epsilon = 2.5e-2
        self.max_rate = 1e-4
        self.lambda_coef = 0 
        self.MLP = nn.Linear(input_dim, 4)
    
    def forward(self, x):
        gamma, nu, alpha, beta = torch.split(self.MLP(x), 1, dim=1)
        nu, alpha, beta = fun.softplus(nu), fun.softplus(alpha)+1, fun.softplus(beta)
        return gamma, nu, alpha, beta
    
    def get_loss(self, y, gamma, nu, alpha, beta):
        nig_loss, nig_regularization = NIG_NLL(y, gamma, nu, alpha, beta), NIG_Regularization(y, gamma, nu, alpha)
        loss = nig_loss + (nig_regularization - self.epsilon)*self.lambda_coef
        return loss, nig_loss, nig_regularization
    
    def hyperparams_update(self, nig_regularization):
        with torch.no_grad():
            self.lambda_coef += self.max_rate * (nig_regularization - self.epsilon)
