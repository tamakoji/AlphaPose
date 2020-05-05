from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path

import numpy as np
import torch

try:
    from torch.utils.cpp_extension import load
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError(
        "The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.join(this_dir, "src")

    sources_main = glob.glob(os.path.join(this_dir, "nms*.cpp"))
    sources_cuda = glob.glob(os.path.join(this_dir, "nms*.cu"))

    sources = sources_main + sources_cuda

    extra_cflags = []
    extra_cuda_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extra_cflags = ["-O3", "-DWITH_CUDA"]
        extra_cuda_cflags = ["--expt-extended-lambda"]
    sources = [os.path.join(this_dir, s) for s in sources]
    extra_include_paths = [this_dir]
    return load(
        name="ap_nms_ext_lib",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
    )

_backend = _load_extensions()


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets.to('cpu')
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = _backend.nms_cuda(dets_th, iou_thr)
        else:
            inds = _backend.nms_cpu(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


"""
todo : cython pyx to native c? but where below code being used??

def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
"""
