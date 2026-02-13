import torch
import torch.nn as nn

class Helmholtz_Loss(nn.Module):
    def __init__(self, wavenumber_k, grid_spacing_l):
        super().__init__()
        
        self.k = wavenumber_k
        self.l = grid_spacing_l

        l_m = [grid_spacing_l**(i+1) for i in range(7)]
        
        self.c1_matrix = torch.tensor([
            [l_m[0],       l_m[1]/2,     l_m[2]/3,     l_m[3]/4],
            [l_m[1]/2,     l_m[2]/3,     l_m[3]/4,     l_m[4]/5],
            [l_m[2]/3,     l_m[3]/4,     l_m[4]/5,     l_m[5]/6],
            [l_m[3]/4,     l_m[4]/5,     l_m[5]/6,     l_m[6]/7]
        ], dtype=torch.float32)

        self.c2_matrix = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4*l_m[0], 6*l_m[1]],
            [0, 0, 6*l_m[1], 12*l_m[2]]
        ], dtype=torch.float32)
        
        
        self.c3_matrix = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2*l_m[0],   l_m[1],     (2/3)*l_m[2], (1/2)*l_m[3]],
            [3*l_m[1],   2*l_m[2],   (3/2)*l_m[3], (6/5)*l_m[4]]
        ], dtype=torch.float32)

        self.m_matrix = torch.tensor([
            [1,          0,           0,           0],         
            [0,          0,           1,           0],          
            [-3/l_m[1],  3/l_m[1],   -2/l_m[0],   -1/l_m[0]],   
            [2/l_m[2],   -2/l_m[2],   1/l_m[1],    1/l_m[1]]    
        ], dtype=torch.float32)

        # Register buffers so they save with the model but don't train
        self.register_buffer('C1', self.c1_matrix)
        self.register_buffer('C2', self.c2_matrix) 
        self.register_buffer('C3', self.c3_matrix)
        self.register_buffer('M', self.m_matrix) 
    
    def forward(self, x):
        #we're going to slice up x in a very explicit way to make the math 
        # work out well 

        #first I setup constants to make the slice more readable is to use the shape
        # with some constants 

        B, C, H, W = x.shape

        BATCH = slice(None)
        CHANNEL = slice(None)

        #Each tensor slice needs start and end points I declare them here 
        start = 0
        end = H-1 
        plus1_start = 1
        plus1_end = H 

        # Explicitly state the top, bottom, left, right 
        top_rows = slice(start, end)
        bottom_rows = slice(plus1_start, plus1_end)
        left_cols = slice(start, end)
        right_cols = slice(plus1_start, plus1_end)

        # Index slices explictly stated 

        x_top_left = x[BATCH, CHANNEL, top_rows, left_cols]
        x_top_right = x[BATCH, CHANNEL, top_rows, right_cols]
        x_bottom_left = x[BATCH, CHANNEL, bottom_rows, left_cols]
        x_bottom_right = x[BATCH, CHANNEL, bottom_rows, right_cols]

        # now we're doing some crazy tensor slicing to build a hermite matrix (equation 3 in the paper)
        # u is the pressure field
        # du_x is the slope in the x direction 
        # du_y is the slope in the y direction 
        # du_xy is the slope in the x and y direction 

        #slice the parts of each index that need to be stacked together
        
        #pressure field values
        u_tl = x_top_left[:, 0] #i hate this indexing method but it will be seven hundred lines if I state this explicitly 
        u_tr = x_top_right[:,0]
        u_bl = x_bottom_left[:,0]
        u_br = x_bottom_right[:,0]

        #x slopes 
        du_x_tl = x_top_left[:, 1] #i hate this indexing method but it will be seven hundred lines if I state this explicitly 
        du_x_tr = x_top_right[:,1]
        du_x_bl = x_bottom_left[:,1]
        du_x_br = x_bottom_right[:,1]

        #y slopes
        du_y_tl = x_top_left[:, 2] #i hate this indexing method but it will be seven hundred lines if I state this explicitly 
        du_y_tr = x_top_right[:,2]
        du_y_bl = x_bottom_left[:,2]
        du_y_br = x_bottom_right[:,2]

        #dxdy twist
        du_xy_tl = x_top_left[:, 3] #i hate this indexing method but it will be seven hundred lines if I state this explicitly 
        du_xy_tr = x_top_right[:,3]
        du_xy_bl = x_bottom_left[:,3]
        du_xy_br = x_bottom_right[:,3]

        #now we stack all the tensors 
        row_1 = torch.stack([u_tl,u_tr,du_y_tl, du_y_tr], dim = -1)
        row_2 = torch.stack([u_bl, u_br, du_y_bl, du_y_br], dim=-1)
        row_3 = torch.stack([du_xy_tl, du_x_tr, du_xy_tl, du_xy_tr], dim=-1)
        row_4 = torch.stack([du_x_bl, du_x_br, du_xy_bl, du_xy_br])

        hermitian_matrix = torch.stack([row_1, row_2, row_3, row_4], dim=-2)

        #now we expand our dimensions to match so that matmul doesn't freak out 
        m_expanded = self.m_matrix.view(1,1,1,4,4)
        m_q = torch.matmul(m_expanded, hermitian_matrix)

        a_matrix = torch.matmul(m_q, m_expanded.transpose(-1,-2))
        a_matrix_transpose = a_matrix.transpose(-1,-2)
        # intermediate calculations 
        
        # Note: A_matrix shape is [Batch, N, 4, 4]
        
        # TERM 1
        # 1a: A @ C1
        intermediate_term_1a = torch.matmul(a_matrix, self.c1_matrix) 
        
        intermediate_term_1b = torch.sum(self.c2_matrix) 
        
        intermediate_term_1c = torch.matmul(intermediate_term_1a, a_matrix_transpose)
        
        term_1 = intermediate_term_1b * torch.sum(intermediate_term_1c, dim=(-1, -2))

        # TERM 2
        intermediate_term_2a = torch.matmul(a_matrix, self.c2_matrix)
        intermediate_term_2b = torch.sum(self.c1_matrix)
        intermediate_term_2c = torch.matmul(intermediate_term_2a, a_matrix_transpose)
        term_2 = intermediate_term_2b * torch.sum(intermediate_term_2c, dim=(-1, -2))

        # TERM 3
        intermediate_term_3a = torch.matmul(a_matrix, self.c1_matrix)
        intermediate_term_3b = torch.sum(self.c1_matrix) 
        intermediate_term_3c = torch.matmul(intermediate_term_3a, a_matrix_transpose)
        term_3 = (self.k**4) * intermediate_term_3b * torch.sum(intermediate_term_3c, dim=(-1, -2))

        # TERM 4
        intermediate_term_4a = torch.matmul(a_matrix, self.c3_matrix.transpose(-1,-2))
        intermediate_term_4b = torch.sum(self.c3_matrix)
        intermediate_term_4c = torch.matmul(intermediate_term_4a, a_matrix_transpose)
        term_4 = 2 * intermediate_term_4b * torch.sum(intermediate_term_4c, dim=(-1, -2))

        # TERM 5
        intermediate_term_5a = torch.matmul(a_matrix, self.c3_matrix)
        intermediate_term_5b = torch.sum(self.c1_matrix)
        intermediate_term_5c = torch.matmul(intermediate_term_5a, a_matrix_transpose)
        term_5 = 2 * (self.k**2) * intermediate_term_5b * torch.sum(intermediate_term_5c, dim=(-1, -2))

        # TERM 6
        intermediate_term_6a = torch.matmul(a_matrix, self.c1_matrix)
        intermediate_term_6b = torch.sum(self.c3_matrix.transpose(-1,-2)) 
        intermediate_term_6c = torch.matmul(intermediate_term_6a, a_matrix_transpose)
        term_6 = 2 * (self.k**2) * intermediate_term_6b * torch.sum(intermediate_term_6c, dim=(-1, -2))

        # Sum all terms to get the integral value for each patch
        patch_integrals = term_1 + term_2 + term_3 + term_4 + term_5 + term_6
        
        # Final result should be the mean over the batch
        return torch.mean(patch_integrals)


        
