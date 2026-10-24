"""Triton kernels for the segment-wise parts of the fast KcatNet: one pass over each edge or row segment, no atomics except in `edge_hidden`."""

import torch
import triton
import triton.language as tl

BLOCK = 64
_DTYPES = {torch.float32: tl.float32, torch.float64: tl.float64}


@triton.jit
def _forward_kernel(
    x_ptr, ptr_ptr, out_ptr, arg_ptr, num_nodes, width, BLOCK: tl.constexpr, DTYPE: tl.constexpr
):
    node = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    start = tl.load(ptr_ptr + node)
    end = tl.load(ptr_ptr + node + 1)
    total = tl.zeros([BLOCK], DTYPE)
    total_sq = tl.zeros([BLOCK], DTYPE)
    low = tl.full([BLOCK], float("inf"), DTYPE)
    high = tl.full([BLOCK], float("-inf"), DTYPE)
    arg_low = tl.zeros([BLOCK], tl.int64)
    arg_high = tl.zeros([BLOCK], tl.int64)
    for edge in range(start, end):
        x = tl.load(x_ptr + edge * width + cols, mask=mask, other=0.0)
        total += x
        total_sq += x * x
        is_low = x < low
        low = tl.where(is_low, x, low)
        arg_low = tl.where(is_low, edge, arg_low)
        is_high = x > high
        high = tl.where(is_high, x, high)
        arg_high = tl.where(is_high, edge, arg_high)
    row = node * width + cols
    step = num_nodes * width
    tl.store(out_ptr + row, total, mask=mask)
    tl.store(out_ptr + step + row, total_sq, mask=mask)
    tl.store(out_ptr + 2 * step + row, low, mask=mask)
    tl.store(out_ptr + 3 * step + row, high, mask=mask)
    tl.store(arg_ptr + row, arg_low, mask=mask)
    tl.store(arg_ptr + step + row, arg_high, mask=mask)


@triton.jit
def _backward_kernel(
    x_ptr,
    ptr_ptr,
    grad_ptr,
    arg_ptr,
    out_ptr,
    num_nodes,
    width,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    node = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    start = tl.load(ptr_ptr + node)
    end = tl.load(ptr_ptr + node + 1)
    row = node * width + cols
    step = num_nodes * width
    grad_total = tl.load(grad_ptr + row, mask=mask, other=0.0)
    grad_total_sq = tl.load(grad_ptr + step + row, mask=mask, other=0.0)
    grad_low = tl.load(grad_ptr + 2 * step + row, mask=mask, other=0.0)
    grad_high = tl.load(grad_ptr + 3 * step + row, mask=mask, other=0.0)
    arg_low = tl.load(arg_ptr + row, mask=mask, other=-1)
    arg_high = tl.load(arg_ptr + step + row, mask=mask, other=-1)
    for edge in range(start, end):
        x = tl.load(x_ptr + edge * width + cols, mask=mask, other=0.0)
        grad = grad_total + 2.0 * x * grad_total_sq
        grad += tl.where(arg_low == edge, grad_low, 0.0)
        grad += tl.where(arg_high == edge, grad_high, 0.0)
        tl.store(out_ptr + edge * width + cols, grad, mask=mask)


class _SegmentStats(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ptr):
        num_nodes, width = ptr.numel() - 1, x.size(1)
        out = torch.empty(4, num_nodes, width, device=x.device, dtype=x.dtype)
        arg = torch.empty(2, num_nodes, width, device=x.device, dtype=torch.int64)
        grid = (num_nodes, triton.cdiv(width, BLOCK))
        _forward_kernel[grid](
            x, ptr, out, arg, num_nodes, width, BLOCK=BLOCK, DTYPE=_DTYPES[x.dtype]
        )
        ctx.save_for_backward(x, ptr, arg)
        return out.unbind(0)

    @staticmethod
    def backward(ctx, *grads):
        x, ptr, arg = ctx.saved_tensors
        num_nodes, width = ptr.numel() - 1, x.size(1)
        out = torch.empty_like(x)
        grid = (num_nodes, triton.cdiv(width, BLOCK))
        _backward_kernel[grid](
            x,
            ptr,
            torch.stack(grads).contiguous(),
            arg,
            out,
            num_nodes,
            width,
            BLOCK=BLOCK,
            DTYPE=_DTYPES[x.dtype],
        )
        return out, None


def segment_stats(x, ptr):
    """Return (sum, sum of squares, min, max) of the rows of `x` in each segment `ptr[i]:ptr[i + 1]`; all segments are non-empty."""
    return _SegmentStats.apply(x.contiguous(), ptr)


@triton.jit
def _hidden_forward_kernel(
    ni_ptr, nj_ptr, edge_ptr, src_ptr, ptr_ptr, out_ptr, width, BLOCK: tl.constexpr
):
    node = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    base = tl.load(ni_ptr + node * width + cols, mask=mask, other=0.0)
    for edge in range(tl.load(ptr_ptr + node), tl.load(ptr_ptr + node + 1)):
        other = tl.load(nj_ptr + tl.load(src_ptr + edge) * width + cols, mask=mask, other=0.0)
        pre = base + other + tl.load(edge_ptr + edge * width + cols, mask=mask, other=0.0)
        tl.store(out_ptr + edge * width + cols, tl.maximum(pre, 0.0), mask=mask)


@triton.jit
def _hidden_backward_kernel(
    grad_ptr,
    out_ptr,
    src_ptr,
    ptr_ptr,
    grad_pre_ptr,
    grad_ni_ptr,
    grad_nj_ptr,
    width,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    node = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    total = tl.zeros([BLOCK], DTYPE)
    for edge in range(tl.load(ptr_ptr + node), tl.load(ptr_ptr + node + 1)):
        grad = tl.load(grad_ptr + edge * width + cols, mask=mask, other=0.0)
        grad = tl.where(
            tl.load(out_ptr + edge * width + cols, mask=mask, other=0.0) > 0.0, grad, 0.0
        )
        tl.store(grad_pre_ptr + edge * width + cols, grad, mask=mask)
        tl.atomic_add(grad_nj_ptr + tl.load(src_ptr + edge) * width + cols, grad, mask=mask)
        total += grad
    tl.store(grad_ni_ptr + node * width + cols, total, mask=mask)


class _EdgeHidden(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_i, node_j, edge, src, ptr):
        num_nodes, width = node_i.shape
        out = torch.empty_like(edge)
        _hidden_forward_kernel[(num_nodes, triton.cdiv(width, BLOCK))](
            node_i, node_j, edge, src, ptr, out, width, BLOCK=BLOCK
        )
        ctx.save_for_backward(out, src, ptr)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, src, ptr = ctx.saved_tensors
        num_nodes, width = ptr.numel() - 1, out.size(1)
        grad_pre = torch.empty_like(out)
        grad_i = torch.empty(num_nodes, width, device=out.device, dtype=out.dtype)
        grad_j = torch.zeros_like(grad_i)
        _hidden_backward_kernel[(num_nodes, triton.cdiv(width, BLOCK))](
            grad.contiguous(),
            out,
            src,
            ptr,
            grad_pre,
            grad_i,
            grad_j,
            width,
            BLOCK=BLOCK,
            DTYPE=_DTYPES[out.dtype],
        )
        return grad_i, grad_j, grad_pre, None, None


def edge_hidden(node_i, node_j, edge, src, ptr):
    """relu(node_i[dst] + node_j[src] + edge) for edges sorted by dst, where `ptr` holds each target node's edge range."""
    return _EdgeHidden.apply(node_i.contiguous(), node_j.contiguous(), edge.contiguous(), src, ptr)


@triton.jit
def _graph_norm_forward_kernel(
    x_ptr,
    ptr_ptr,
    scale_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    mean_ptr,
    std_ptr,
    eps_ptr,
    width,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    graph = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    start = tl.load(ptr_ptr + graph)
    end = tl.load(ptr_ptr + graph + 1)
    count = tl.maximum(end - start, 1).to(DTYPE)
    total = tl.zeros([BLOCK], DTYPE)
    for row in range(start, end):
        total += tl.load(x_ptr + row * width + cols, mask=mask, other=0.0)
    mean = total / count
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    total_sq = tl.zeros([BLOCK], DTYPE)
    for row in range(start, end):
        centered = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0) - mean * scale
        total_sq += centered * centered
    std = tl.sqrt(total_sq / count + tl.load(eps_ptr))
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    for row in range(start, end):
        centered = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0) - mean * scale
        tl.store(y_ptr + row * width + cols, weight * centered / std + bias, mask=mask)
    tl.store(mean_ptr + graph * width + cols, mean, mask=mask)
    tl.store(std_ptr + graph * width + cols, std, mask=mask)


@triton.jit
def _graph_norm_backward_kernel(
    grad_ptr,
    x_ptr,
    ptr_ptr,
    scale_ptr,
    weight_ptr,
    mean_ptr,
    std_ptr,
    dx_ptr,
    dparam_ptr,
    num_graphs,
    width,
    BLOCK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    graph = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    start = tl.load(ptr_ptr + graph)
    end = tl.load(ptr_ptr + graph + 1)
    count = tl.maximum(end - start, 1).to(DTYPE)
    mean = tl.load(mean_ptr + graph * width + cols, mask=mask, other=0.0)
    std = tl.load(std_ptr + graph * width + cols, mask=mask, other=1.0)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    sum_grad = tl.zeros([BLOCK], DTYPE)
    sum_grad_centered = tl.zeros([BLOCK], DTYPE)
    sum_centered = tl.zeros([BLOCK], DTYPE)
    for row in range(start, end):
        grad = tl.load(grad_ptr + row * width + cols, mask=mask, other=0.0)
        centered = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0) - mean * scale
        sum_grad += grad
        sum_grad_centered += grad * centered
        sum_centered += centered
    grad_var = -weight * sum_grad_centered / (2.0 * std * std * std)
    sum_grad_c = weight / std * sum_grad + grad_var * 2.0 / count * sum_centered
    for row in range(start, end):
        grad = tl.load(grad_ptr + row * width + cols, mask=mask, other=0.0)
        centered = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0) - mean * scale
        grad_c = grad * weight / std + grad_var * 2.0 * centered / count
        tl.store(dx_ptr + row * width + cols, grad_c - scale * sum_grad_c / count, mask=mask)
    step = num_graphs * width
    tl.store(dparam_ptr + graph * width + cols, sum_grad_centered / std, mask=mask)
    tl.store(dparam_ptr + step + graph * width + cols, sum_grad, mask=mask)
    tl.store(dparam_ptr + 2 * step + graph * width + cols, -mean * sum_grad_c, mask=mask)


class _GraphNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ptr, scale, weight, bias, eps):
        num_graphs, width = ptr.numel() - 1, x.size(1)
        y = torch.empty_like(x)
        mean = torch.empty(num_graphs, width, device=x.device, dtype=x.dtype)
        std = torch.empty_like(mean)
        grid = (num_graphs, triton.cdiv(width, BLOCK))
        _graph_norm_forward_kernel[grid](
            x,
            ptr,
            scale,
            weight,
            bias,
            y,
            mean,
            std,
            x.new_tensor(eps),
            width,
            BLOCK=BLOCK,
            DTYPE=_DTYPES[x.dtype],
        )
        ctx.save_for_backward(x, ptr, scale, weight, mean, std)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, ptr, scale, weight, mean, std = ctx.saved_tensors
        num_graphs, width = ptr.numel() - 1, x.size(1)
        dx = torch.empty_like(x)
        dparam = torch.empty(3, num_graphs, width, device=x.device, dtype=x.dtype)
        grid = (num_graphs, triton.cdiv(width, BLOCK))
        _graph_norm_backward_kernel[grid](
            grad.contiguous(),
            x,
            ptr,
            scale,
            weight,
            mean,
            std,
            dx,
            dparam,
            num_graphs,
            width,
            BLOCK=BLOCK,
            DTYPE=_DTYPES[x.dtype],
        )
        d_weight, d_bias, d_scale = dparam.sum(1)
        return dx, None, d_scale, d_weight, d_bias, None


def graph_norm(gn, x, ptr):
    """GraphNorm of `gn` for rows grouped into the contiguous graphs `ptr[i]:ptr[i + 1]`."""
    return _GraphNorm.apply(x.contiguous(), ptr, gn.mean_scale, gn.weight, gn.bias, gn.eps)
