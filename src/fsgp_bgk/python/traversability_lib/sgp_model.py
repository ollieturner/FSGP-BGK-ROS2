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
    # Added: lengthscale may be one value (all inputs) or one per input [x, y, curvature, gradient];
    # lengthscale_max: optional per-input upper bounds (None or a None entry = unbounded), apply_kernel_init only
    def __init__(self, train_x, train_y, likelihood, inducing_points, lengthscale=0.7, alpha=10, apply_kernel_init=False,
                 lengthscale_max=None):
        super(SGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # TODO Choose a subset of training points to make it actually sparse (random or k-means) - or compute earlier with inducing_points
        inducing_variable=train_x

        # Choose gpytorch.kernels.RQKernel or gpytorch.kernels.RBFKernel
        
        # RQKernel:
        # Added apply_kernel_init argument to switch between new and original behaviour
        if apply_kernel_init:
            # New: Properly use inputted lengthscale and alpha
            # One lengthscale per input (ARD), each starting at its given value and moving separately during training
            n_dims = train_x.shape[-1]
            init = torch.as_tensor(lengthscale, dtype=torch.float32).expand(n_dims).clone()
            constraint = None
            if lengthscale_max is not None:
                # Upper bounds per input (unbounded = 1e4); the starting value must lie inside the bound
                upper = torch.tensor([1e4 if m is None else float(m) for m in lengthscale_max], dtype=torch.float32)
                constraint = gpytorch.constraints.Interval(torch.full_like(upper, 1e-4), upper)
                init = torch.minimum(init, 0.95 * upper)
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=n_dims, lengthscale_constraint=constraint)
            rq_kernel.lengthscale = init.reshape(1, n_dims)
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