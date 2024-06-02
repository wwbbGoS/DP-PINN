import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class AC_PINN():
    def __init__(self,x_u, y_u, x_f,x_lb,x_ub, net, device, nepochs, lambda_val, noise = 0.0):
        super(AC_PINN, self).__init__()

        self.x_u = x_u
        self.x_f = x_f
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.net = net

        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=1e-4, betas = (0.9, 0.999))
        # new #
        self.net_optim2 = torch.optim.LBFGS(self.net.parameters(), lr = 1e-4)
        
        self.device = device


        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        #Boundary conditions
        self.train_x_lb = torch.tensor(x_lb, requires_grad=True).float().to(device)
        self.train_x_ub = torch.tensor(x_ub, requires_grad=True).float().to(device)


        self.nepochs = nepochs
        self.lambda_val = lambda_val
        
        self.batch_size = 32

        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )  
    
    def get_residual(self, X):
        # physics loss for collocation/boundary points
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        u = self.net.forward(torch.cat([x, t], dim=1))
        f = self.phy_residual(x, t, u)
        return u, f

    def boundary_loss(self):
        x_lb = torch.tensor(self.train_x_lb[:, 0:1], requires_grad=True).float().to(self.device)
        t_lb = torch.tensor(self.train_x_lb[:, 1:2], requires_grad=True).float().to(self.device)

        x_ub = torch.tensor(self.train_x_ub[:, 0:1], requires_grad=True).float().to(self.device)
        t_ub = torch.tensor(self.train_x_ub[:, 1:2], requires_grad=True).float().to(self.device)
        
        y_pred_lb = self.net.forward(torch.cat([x_lb, t_lb], dim=1))
        y_pred_ub = self.net.forward(torch.cat([x_ub, t_ub], dim=1))
        
        u_lb = y_pred_lb[:,0:1]
        u_ub = y_pred_ub[:,0:1]

        u_lb_x = torch.autograd.grad(
                u_lb, x_lb, 
                grad_outputs=torch.ones_like(u_lb),
                retain_graph=True,
                create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
                u_ub, x_ub, 
                grad_outputs=torch.ones_like(u_ub),
                retain_graph=True,
                create_graph=True
            )[0]

        loss = torch.mean((y_pred_lb - y_pred_ub)**2) + \
               torch.mean((u_lb_x - u_ub_x)**2)
        return loss

    def phy_residual(self, x, t, u, nu = (0.01/np.pi)):
        """ The pytorch autograd version of calculating residual """

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - 0.0001* u_xx  + 5*u*u*u - 5*u 
        return f

    
#     def train(self,phase1,trunk1_X,trunk1_u):
#         TOT_loss = np.zeros(self.nepochs)
#         MSE_loss = np.zeros(self.nepochs)
#         PHY_loss = np.zeros(self.nepochs)

#         for epoch in range(self.nepochs):
#             epoch_loss = 0
#             for i, (x, y) in enumerate(self.train_loader):

#                 self.net_optim.zero_grad()
#                 y_pred, _ = self.get_residual(x)
#                 _, f_u  = self.get_residual(self.train_x_f)

#                 mse_loss = torch.nn.functional.mse_loss(y, y_pred)
#                 physics_loss = torch.mean(f_u**2)
                
#                 b_loss = self.boundary_loss()

#                 if(phase1):
#                     loss = 100 * (mse_loss) + self.lambda_val * physics_loss  + b_loss
#                 else:
#                     loss = 100 * (mse_loss) + self.lambda_val * physics_loss  + b_loss
                    
#                 loss.backward(retain_graph=True)
#                 self.net_optim.step()


#                 TOT_loss[epoch] += loss.detach().cpu().numpy()
#                 MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
#                 PHY_loss[epoch] += physics_loss.detach().cpu().numpy()


#             TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)
#             MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
#             PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)

            
#             if (epoch % 100 == 0):
#                 print(
#                     "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
#                     % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
#                 )
            

#             if (epoch % 2500 == 2499):
# #                 print(
# #                     "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
# #                     % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
# #                 )
                
                
#                 u_pred_list = []
#                 f_u_pred_list = []

#                 for run in range(1):
#                     y_pred, f_u_pred = self.get_residual(trunk1_X)
#                     u_pred = y_pred[:,0:1].detach().cpu().numpy()
#                     u_pred_list.append(u_pred)
#                     f_u_pred_list.append(f_u_pred.detach().cpu().numpy())


#                 u_pred_arr = np.array(u_pred_list)
#                 f_u_pred_arr = np.array(f_u_pred_list)

#                 u_pred = u_pred_arr.mean(axis=0)
#                 f_u_pred = f_u_pred_arr.mean(axis=0)

#                 residual = (f_u_pred**2).mean()
#                 error_u = np.linalg.norm(trunk1_u-u_pred,2)/np.linalg.norm(trunk1_u,2)
                
#                 tmp_pred = u_pred[-512:]
                
#                 ts_error = np.linalg.norm(trunk1_u[-512:]-u_pred[-512:],2)/np.linalg.norm(trunk1_u[-512:],2)
                
#                 print(
#                     "[Epoch %d/%d] [MSE Error %f] [Residual Error %f] [ts error %f] "
#                     % (epoch, self.nepochs, error_u, residual, ts_error)
#                 )
                
    
    ## test section ###
    
    
    def train(self,phase1,trunk1_X,trunk1_u):
        TOT_loss = np.zeros(self.nepochs)
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)

        for epoch in range(self.nepochs):
            for i, (x, y) in enumerate(self.train_loader):

                self.net_optim.zero_grad()
                y_pred, _ = self.get_residual(x)
                _, f_u  = self.get_residual(self.train_x_f)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(f_u**2)
                
                b_loss = self.boundary_loss()

                if(phase1):
                    loss = 100 * (mse_loss) + self.lambda_val * physics_loss  + b_loss
                else:
                    loss = 100 * (mse_loss) + self.lambda_val * physics_loss  + b_loss
                    
                loss.backward(retain_graph=True)
                self.net_optim.step()


                TOT_loss[epoch] += loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += physics_loss.detach().cpu().numpy()


            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)
            
            
            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
                )

        for epoch in range(self.nepochs):
            if (epoch % 2500 == 2499):
                print(epoch)
            for i, (x, y) in enumerate(self.train_loader):
                self.net_optim2.zero_grad()
                
                def closure():
                    y_pred, _ = self.get_residual(x)
                    _, f_u  = self.get_residual(self.train_x_f)
                    mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                    physics_loss = torch.mean(f_u**2)
                    b_loss = self.boundary_loss()

                    loss = (mse_loss) + self.lambda_val * physics_loss  + b_loss
                    loss.backward()
                    return loss
                
                self.net_optim2.step(closure)
            
             
                
            
#             if (epoch % 100 == 0):
#                 print(
#                     "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
#                     % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
#                 )
            

#             if (epoch % 2500 == 2499):
#                 print(
#                     "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
#                     % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
#                 )
                
                
#                 u_pred_list = []
#                 f_u_pred_list = []

#                 for run in range(1):
#                     y_pred, f_u_pred = self.get_residual(trunk1_X)
#                     u_pred = y_pred[:,0:1].detach().cpu().numpy()
#                     u_pred_list.append(u_pred)
#                     f_u_pred_list.append(f_u_pred.detach().cpu().numpy())


#                 u_pred_arr = np.array(u_pred_list)
#                 f_u_pred_arr = np.array(f_u_pred_list)

#                 u_pred = u_pred_arr.mean(axis=0)
#                 f_u_pred = f_u_pred_arr.mean(axis=0)

#                 residual = (f_u_pred**2).mean()
#                 error_u = np.linalg.norm(trunk1_u-u_pred,2)/np.linalg.norm(trunk1_u,2)
                
#                 tmp_pred = u_pred[-512:]
                
#                 ts_error = np.linalg.norm(trunk1_u[-512:]-u_pred[-512:],2)/np.linalg.norm(trunk1_u[-512:],2)
                
#                 print(
#                     "[Epoch %d/%d] [MSE Error %f] [Residual Error %f] [ts error %f] "
#                     % (epoch, self.nepochs, error_u, residual, ts_error)
#                 )
class PINN():
    def __init__(self,x_u, y_u, x_f,x_lb,x_ub, net, device, nepochs, lambda_val, noise = 0.0):
        super(PINN, self).__init__()

        self.x_u = x_u
        self.x_f = x_f
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.net = net

        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=1e-4, betas = (0.9, 0.999))
        
        self.device = device


        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        #Boundary conditions
        self.train_x_lb = torch.tensor(x_lb, requires_grad=True).float().to(device)
        self.train_x_ub = torch.tensor(x_ub, requires_grad=True).float().to(device)


        self.nepochs = nepochs
        self.lambda_val = lambda_val
        
        self.batch_size = 15000

        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )  
    
    def get_residual(self, X):
        # physics loss for collocation/boundary points
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        u = self.net.forward(torch.cat([x, t], dim=1))
        f = self.phy_residual(x, t, u)
        return u, f

    def boundary_loss(self):
        x_lb = torch.tensor(self.train_x_lb[:, 0:1], requires_grad=True).float().to(self.device)
        t_lb = torch.tensor(self.train_x_lb[:, 1:2], requires_grad=True).float().to(self.device)

        x_ub = torch.tensor(self.train_x_ub[:, 0:1], requires_grad=True).float().to(self.device)
        t_ub = torch.tensor(self.train_x_ub[:, 1:2], requires_grad=True).float().to(self.device)
        
        y_pred_lb = self.net.forward(torch.cat([x_lb, t_lb], dim=1))
        y_pred_ub = self.net.forward(torch.cat([x_ub, t_ub], dim=1))
        
        u_lb = y_pred_lb[:,0:1]
        u_ub = y_pred_ub[:,0:1]

        u_lb_x = torch.autograd.grad(
                u_lb, x_lb, 
                grad_outputs=torch.ones_like(u_lb),
                retain_graph=True,
                create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
                u_ub, x_ub, 
                grad_outputs=torch.ones_like(u_ub),
                retain_graph=True,
                create_graph=True
            )[0]

        loss = torch.mean((y_pred_lb - y_pred_ub)**2) + \
               torch.mean((u_lb_x - u_ub_x)**2)
        return loss

    def phy_residual(self, x, t, u, nu = (0.01/np.pi)):
        """ The pytorch autograd version of calculating residual """

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t - 0.0001* u_xx  + 5*u*u*u - 5*u 
        return f

    def train(self):
        TOT_loss = np.zeros(self.nepochs)
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)

        for epoch in range(self.nepochs):
            for i, (x, y) in enumerate(self.train_loader):

                self.net_optim.zero_grad()
                y_pred, _ = self.get_residual(x)
                _, f_u  = self.get_residual(self.train_x_f)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(f_u**2)
                
                b_loss = self.boundary_loss()

                loss =  1 * (mse_loss ) + self.lambda_val * physics_loss + b_loss

                loss.backward(retain_graph=True)
                self.net_optim.step()


                TOT_loss[epoch] += loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += physics_loss.detach().cpu().numpy()


            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)


            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch])
                )