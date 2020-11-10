CUDA Kernel Programming Guide
=============================

Performance Guide
^^^^^^^^^^^^^^^^^
Very important and useful. Follow the `CUDA Documentation <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>`_ instead of other sources.


Coalesced Access to Global Memory
"""""""""""""""""""""""""""""""""
`Coalesced Access to Global Memory <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory>`_

- | Refer OverlayKernel.cu and EffectsKernel.cu 
- | uchar4 (4 bytes) - 32x32 threads per block - 4x32x32 - 4K bytes 
- | A big difference - like 2x in Performance

Math Library
""""""""""""
`NVIDIA CUDA Math API <https://docs.nvidia.com/cuda/cuda-math-api/index.html>`_

- | multiplication use from here 
- | big difference 

__device__ functions 
""""""""""""""""""""
For writing clean/reusable code, I was using __device__ function - but the Performance dropped by half. So, I started using macros. I didn't investigate more on why?