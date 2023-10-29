/*
 * @Author: BigCiLeng && bigcileng@outlook.com
 * @Date: 2023-10-27 19:49:26
 * @LastEditors: BigCiLeng && bigcileng@outlook.com
 * @LastEditTime: 2023-10-29 18:35:43
 * @FilePath: /cuda_cxx_extension/interpolation.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
 */
#include <torch/extension.h>
#include "utils.h"

torch::Tensor trilinear_interpolation(
    const torch::Tensor feats,
    const torch::Tensor points
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_forward_cu(feats, points);
}

torch::Tensor trilinear_interpolation_backward(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
){
    CHECK_INPUT(dL_dfeat_interp);
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_backward_cu(dL_dfeat_interp, feats, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
    m.def("trilinear_interpolation_backward", &trilinear_interpolation_backward);
}