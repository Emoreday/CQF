#pragma once
#include <torch/extension.h>

at::Tensor mapping_dist_324_gpu(
  const at::Tensor &src_tensor,
  const at::Tensor &index_tensor,
//  at::Tensor &dist_tensor,
  const at::Tensor shape_src_tensor,
  const at::Tensor shape_index_tensor,
  const at::Tensor shape_target_tensor
  );

at::Tensor mapping_src_223_gpu(
  const at::Tensor &src_tensor,
  const at::Tensor &index_tensor,
//  at::Tensor &dist_tensor,
  const at::Tensor shape_src_tensor,
  const at::Tensor shape_index_tensor,
  const at::Tensor shape_target_tensor
  );

at::Tensor add_gpu(const at::Tensor &a_tensor, const at::Tensor &b_tensor);

at::Tensor bfp_quant_gpu(
    const at::Tensor &src,
    const at::Tensor &exponent,
    float emin,
    float emax,
    float max_number,
    float mbit,
    bool stochastic
    );