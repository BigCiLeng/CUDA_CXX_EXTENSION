/*
 * @Author: BigCiLeng && bigcileng@outlook.com
 * @Date: 2023-10-27 19:49:26
 * @LastEditors: BigCiLeng && bigcileng@outlook.com
 * @LastEditTime: 2023-10-28 22:55:10
 * @FilePath: /cuda_cxx_extension/interpolation.cpp
 * @Description: 
 * 
 * Copyright (c) 2023 by bigcileng@outlook.com, All Rights Reserved. 
 */
#include <torch/extension.h>
#include "utils.h"

torch::Tensor trilinear_interpolation(
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);

    return trilinear_forward_cu(feats, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}