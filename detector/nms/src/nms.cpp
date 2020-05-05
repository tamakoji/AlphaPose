#include <torch/extension.h>
#include "nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cpu", &nms_cpu, "non-maximum suppression");
  m.def("nms_cuda", &nms_cuda, "non-maximum suppression");
}