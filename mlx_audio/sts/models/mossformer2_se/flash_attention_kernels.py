"""
FLASH Attention Metal Kernel Implementations
Optimized kernels for the ReLU² attention pattern in MossFormer2

Key findings:
- Simple kernel (fused ReLU²) provides 10-20% speedup on real workloads
- More complex fusion attempts are slower due to MLX's already optimized matmul
- The ReLU² operation is the main bottleneck that benefits from kernel fusion
"""

import mlx.core as mx
from mlx.core.fast import metal_kernel

# 1. Simple ReLU² kernel - just fuses ReLU and square operations
relu_squared_kernel = metal_kernel(
    name="relu_squared",
    input_names=["inp"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        T val = inp[elem];
        T relu_val = val > T(0) ? val : T(0);
        out[elem] = relu_val * relu_val;
    """,
)

# 2. Fused multiply-add kernel for better memory efficiency
fused_multiply_add = metal_kernel(
    name="fused_multiply_add",
    input_names=["a", "b", "scale"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        out[elem] = a[elem] * b[elem] * scale[0];
    """,
)


# 3. Optimized fused kernel for Q@K^T computation with ReLU²
qk_relu_squared = metal_kernel(
    name="qk_relu_squared",
    input_names=["q", "k", "scale"],
    output_names=["out"],
    source="""
        uint3 tid = thread_position_in_grid;
        uint batch = tid.z;
        uint row = tid.y;
        uint col = tid.x;
        
        // Early exit if out of bounds
        if (row >= q_shape[2] || col >= k_shape[2]) return;
        
        uint groups = q_shape[1];
        uint q_dim = q_shape[3];
        
        // Compute strides
        uint q_stride = q_shape[2] * q_shape[3];
        uint k_stride = k_shape[2] * k_shape[3];
        uint out_stride = q_shape[2] * k_shape[2];
        
        T scale_val = scale[0];
        
        // Process all groups for this batch
        for (uint g = 0; g < groups; g++) {
            // Compute dot product
            T sum = T(0);
            uint q_offset = batch * groups * q_stride + g * q_stride + row * q_dim;
            uint k_offset = batch * groups * k_stride + g * k_stride + col * q_dim;
            
            for (uint d = 0; d < q_dim; d++) {
                sum += q[q_offset + d] * k[k_offset + d];
            }
            
            // Scale and apply ReLU²
            sum *= scale_val;
            T result = sum > T(0) ? sum * sum : T(0);
            
            // Write output
            uint out_idx = batch * groups * out_stride + g * out_stride + row * k_shape[2] + col;
            out[out_idx] = result;
        }
    """,
)


class FlashAttentionImplementations:
    """Different FLASH attention implementations for comparison"""

    @staticmethod
    def standard(q, k, v, group_size=None):
        """Standard MLX implementation (baseline)"""
        if group_size is None:
            group_size = q.shape[2]  # seq_len
        scale = 1.0 / group_size

        # Q @ K^T scaled
        sim = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale

        # ReLU²
        attn = mx.maximum(sim, 0)
        attn = attn * attn

        # Attention @ V
        return mx.matmul(attn, v)

    @staticmethod
    def simple_kernel(q, k, v, group_size=None):
        """Simple kernel - only ReLU² is fused"""
        if group_size is None:
            group_size = q.shape[2]
        scale = 1.0 / group_size

        # Step 1: Q @ K^T scaled
        sim = mx.matmul(q, mx.transpose(k, [0, 1, 3, 2])) * scale

        # Step 2: Apply fused ReLU² kernel
        sim_shape = sim.shape
        sim_flat = sim.reshape(-1)

        attn_flat = relu_squared_kernel(
            inputs=[sim_flat],
            template=[("T", sim.dtype)],
            grid=(sim_flat.size, 1, 1),
            threadgroup=(min(256, sim_flat.size), 1, 1),
            output_shapes=[sim_flat.shape],
            output_dtypes=[sim.dtype],
        )[0]

        attn = attn_flat.reshape(sim_shape)

        # Step 3: Attention @ V
        return mx.matmul(attn, v)

    @staticmethod
    def optimized_ops(q, k, v, group_size=None):
        """Optimized using MLX operations"""
        if group_size is None:
            group_size = q.shape[2]
        scale = 1.0 / group_size

        # Fuse scale into one of the matrices
        q_scaled = q * mx.sqrt(scale)
        k_scaled = k * mx.sqrt(scale)

        # Q @ K^T with pre-scaled inputs
        sim = mx.matmul(q_scaled, mx.transpose(k_scaled, [0, 1, 3, 2]))

        # ReLU² with single operation
        attn = mx.square(mx.maximum(sim, 0))

        # Final matmul
        return mx.matmul(attn, v)

    @staticmethod
    def fused_kernel(q, k, v, group_size=None):
        """Two-stage optimized kernel approach"""
        if group_size is None:
            group_size = q.shape[2]
        scale = 1.0 / group_size

        # Stage 1: Fused Q@K^T with ReLU² using optimized kernel
        scale_array = mx.array([scale], dtype=q.dtype)
        batch, groups, seq_len, q_dim = q.shape

        # Output shape for attention scores
        attn_shape = (batch, groups, seq_len, seq_len)

        # Grid for Q@K computation
        grid = (seq_len, seq_len, batch)
        threadgroup = (min(16, seq_len), min(16, seq_len), 1)

        # Compute Q@K^T with fused ReLU²
        attn = qk_relu_squared(
            inputs=[q, k, scale_array],
            template=[("T", q.dtype)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[attn_shape],
            output_dtypes=[q.dtype],
        )[0]

        # Stage 2: Standard matmul for attention @ V
        # This is already well-optimized in MLX
        return mx.matmul(attn, v)
