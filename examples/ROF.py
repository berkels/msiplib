#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float32
from msiplib.io.terminal import Timer
from msiplib.optimization import pd_hybrid_grad_alg
from msiplib.proximal_mappings import proxL2Data, project_unit_ball2D
from msiplib.finite_differences import gradient_FD2D, divergence_FD2D

# Parameter
lambda_ = 0.1
maxIter = 1000
stopEps = 0.001
use_cupy = False

# Read input image
f = img_as_float32(imread('mandril.jpg', as_gray=True))

# To use cupy, we need to convert our numpy array to a cupy array (which copies the data to the GPU)
if use_cupy:
    import cupy as cp
    f = cp.array(f)

# Rudin-Osher-Fatemi-denoising
with Timer():
    fDenoised = pd_hybrid_grad_alg(
        f, lambda u, tau: proxL2Data(u, tau, f), project_unit_ball2D, gradient_FD2D, divergence_FD2D,
        lambda_, maxIter, stopEps, PDHGAlgType=2, gamma=0.7)[0]

# Convert the cupy arrays to numpy arrays (which copies the data from the GPU back to the CPU)
if use_cupy:
    f = cp.asnumpy(f)
    fDenoised = cp.asnumpy(fDenoised)

plt.subplot(1, 2, 1)
plt.imshow(f, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
plt.axis('off')
plt.title('Input image')

plt.subplot(1, 2, 2)
plt.imshow(fDenoised, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
plt.axis('off')
plt.title('ROF result')
plt.show()
