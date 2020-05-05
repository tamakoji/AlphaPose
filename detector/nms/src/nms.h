at::Tensor nms_cuda(const at::Tensor& dets, const float threshold);
at::Tensor nms_cpu(const at::Tensor& dets, const float threshold);
