import torch

def get_rotation_Matrix(alpha, beta, gamma):

    """
    alpha  (B, 1)      : Yaw
    beta   (B, 1)      : Pitch  
    gamma  (B, 1)      : Roll 

    R = R_z(alpha) x R_y(beta) x R_x(gamma)

    return R_Matrix (B,3,3)
    """
    x1 = (torch.cos(alpha).mul(torch.cos(beta)))
    x2 = ((torch.cos(alpha)).mul(torch.sin(beta))).mul(torch.sin(gamma))-(torch.sin(alpha)).mul(torch.cos(gamma))
    x3 = ((torch.cos(alpha)).mul(torch.sin(beta))).mul(torch.cos(gamma))+(torch.sin(alpha)).mul(torch.sin(gamma))
    x4 = (torch.sin(alpha).mul(torch.cos(beta)))
    x5 = ((torch.sin(alpha)).mul(torch.sin(beta))).mul(torch.sin(gamma))+(torch.cos(alpha)).mul(torch.cos(gamma))
    x6 = ((torch.sin(alpha)).mul(torch.sin(beta))).mul(torch.cos(gamma))-(torch.cos(alpha)).mul(torch.sin(gamma))
    x7 = -torch.sin(beta)
    x8 = (torch.cos(beta).mul(torch.sin(gamma)))
    x9 = (torch.cos(beta).mul(torch.cos(gamma)))
    
    row1 = torch.cat((x1, x2, x3), -1).unsqueeze(-1)   #(B,3)
    row2 = torch.cat((x4, x5, x6), -1).unsqueeze(-1)   #(B,3)
    row3 = torch.cat((x7, x8, x9), -1).unsqueeze(-1)   #(B,3)
    R_Matrix = torch.cat((row1, row2, row3),-1).transpose(1,2)       #(B,3,3)
    


    return R_Matrix