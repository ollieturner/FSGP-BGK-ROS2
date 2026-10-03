'''
MIT License

Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import gpytorch

class SGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points, lengthscale=0.7, alpha=10, apply_kernel_init=False):
        super(SGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # TODO Choose a subset of training points to make it actually sparse (random or k-means) - or compute earlier with inducing_points
        inducing_variable=train_x

        # Choose gpytorch.kernels.RQKernel or gpytorch.kernels.RBFKernel
        
        # RQKernel:
        # Added apply_kernel_init argument to switch between new and original behaviour
        if apply_kernel_init:
            # New: Properly use inputted lengthscale and alpha
            # Currently uses same lengthscale for all 4 inputs, but lets them move separately during training
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=train_x.shape[-1])
            rq_kernel.lengthscale = lengthscale # TODO Try starting with different lengthscales: rq_kernel.lengthscale = torch.tensor([lx, ly, lc, lg])
            rq_kernel.alpha = alpha
            self.base_covar_module = gpytorch.kernels.ScaleKernel(rq_kernel)
        else:
            # Original: Actually uses GPyTorch's default (0.693) with a single shared lengthscale
            self.base_covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RQKernel(lengthscale=torch.tensor([lengthscale, lengthscale]), alpha=torch.tensor([alpha]))
            )

        # RBFKernel
        # self.base_covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(lengthscale=torch.tensor([lengthscale, lengthscale]), alpha=torch.tensor([alpha]))
        # )
        
        # Make GP a sparse GP
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=inducing_variable, likelihood=likelihood)
 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze()   
        covar_x = self.covar_module(x)  
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  