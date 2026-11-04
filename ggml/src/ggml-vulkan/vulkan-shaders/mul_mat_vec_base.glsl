#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
    uint glu_op;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
#endif
} p;

#ifdef MUL_MAT_ID
uint expert_id;
bool expert_skip; // an id of -1 skips the slot, the result row is zeroed
#endif

#ifdef MUL_MAT_ID_GLU
uint glu_pass; // 0: gate weights, 1: up weights (see compute_rows)
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    const int expert = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
    expert_skip = expert == -1;
    expert_id   = expert_skip ? 0 : uint(expert); // clamp, so that the offsets below stay in range
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + p.expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + p.expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

#ifdef MUL_MAT_ID_GLU
FLOAT_TYPE glu_gate[NUM_COLS][NUM_ROWS];
#endif

#ifdef MUL_MAT_ID_GLU
// D = act(gate) * up with SWIGLU (glu_op 1) or GEGLU (glu_op 2), the same formulas as swiglu.comp and geglu.comp
void store_output(const uint j, const uint n, const uint32_t idx, const FLOAT_TYPE v) {
    if (glu_pass == 0) {
        glu_gate[j][n] = v;
        return;
    }
    const FLOAT_TYPE g = glu_gate[j][n];
    FLOAT_TYPE act;
    if (p.glu_op == 2) {
        const FLOAT_TYPE val = FLOAT_TYPE(0.79788456080286535587989211986876) * g * (FLOAT_TYPE(1) + FLOAT_TYPE(0.044715) * g * g);
        act = FLOAT_TYPE(0.5) * g * (FLOAT_TYPE(2) - FLOAT_TYPE(2) / (exp(FLOAT_TYPE(2) * val) + FLOAT_TYPE(1)));
    } else {
        act = g / (FLOAT_TYPE(1) + exp(-g));
    }
    data_d[idx] = D_TYPE(act * v);
}

// the gate and up sums of a fused MUL_MAT_ID_GLU row are computed by the same workgroup, one after the other
void compute_outputs(const uint32_t first_row, const uint32_t num_rows);

void compute_rows(const uint32_t first_row, const uint32_t num_rows) {
    glu_pass = 0;
    compute_outputs(first_row, num_rows);
    // the up reduction reuses the shared memory of the gate reduction
    barrier();
    glu_pass = 1;
    compute_outputs(first_row, num_rows);
}
#else
void store_output(const uint j, const uint n, const uint32_t idx, const FLOAT_TYPE v) {
    data_d[idx] = D_TYPE(v);
}

#define compute_rows compute_outputs
#endif

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                store_output(j, n, j*p.batch_stride_d + d_offset + first_row + n, temp[j][n]);
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                store_output(j, n, j*p.batch_stride_d + d_offset + first_row + n, temp[j][n]);
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    tmpsh[j][n][0] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                store_output(j, n, j*p.batch_stride_d + d_offset + first_row + n, tmpsh[j][n][0]);
            }
        }
    }
#endif
}
#endif
