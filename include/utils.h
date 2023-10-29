/*
 * @Author: BigCiLeng && bigcileng@outlook.com
 * @Date: 2023-10-28 22:45:04
 * @LastEditors: BigCiLeng && bigcileng@outlook.com
 * @LastEditTime: 2023-10-29 18:34:32
 * @FilePath: /cuda_cxx_extension/include/utils.h
 * @Description: 
 * 
 * Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
 */
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_forward_cu(
    torch::Tensor feats,
    torch::Tensor points
);

torch::Tensor trilinear_backward_cu(
    const torch::Tensor dL_dfeat_interp,
    torch::Tensor feats,
    torch::Tensor points 
);