#include <torch/extension.h>
#include "ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "add",
        &add_gpu,
        "custom addition for bfp kmeans"
    );
    m.def(
      "bfp_quant",
      &bfp_quant_gpu,
      "custom quant for bfp"
    );
    m.def(
      "mapping_dist_324",
      &mapping_dist_324_gpu,
      "custom mapping for bfp kmeans"
    );
    m.def(
      "mapping_src_223",
      &mapping_src_223_gpu,
      "custom mapping for bfp kmeans"
    );

}
