// CUDA implementation of the free-fermion contraction helpers of
// FreeFermion/linalg.py: the mode-basis vectors of a list of operators, their
// vacuum contraction matrix and its Pfaffian, for a whole batch at a time.
//
// pfaffian() follows the scalar reference of FreeFermion/linalg.py step by step:
// skew-symmetric elimination with two-by-two pivots, Pf(A) = A[k, k+1] *
// Pf(Schur complement), the sign of every pivoting transposition tracked, and a
// matrix whose pivot vanishes and has no candidate gives Pf = 0.  Every pivot
// transposition is applied to the running matrix on its own, exactly like the
// reference swaps the two rows and columns of its working copy.
//
// Build and load through FreeFermion/cuda/fermion.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

namespace {

constexpr int MAX_SIZE = 32;      // largest Pfaffian matrix the kernel supports
constexpr int BLOCK = 64;         // threads per block, one block per matrix

__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ cuDoubleComplex cdiv(cuDoubleComplex a, cuDoubleComplex b) {
    const double norm = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / norm,
                                (a.y * b.x - a.x * b.y) / norm);
}

// One block per matrix: the elimination is done cooperatively in shared memory.
__global__ void pfaffian_kernel(const cuDoubleComplex* __restrict__ matrices,
                                cuDoubleComplex* __restrict__ results,
                                int size, double relative_tolerance) {
    __shared__ cuDoubleComplex work[MAX_SIZE * MAX_SIZE];
    __shared__ double partial[BLOCK];
    __shared__ int candidate_of_pivot;
    __shared__ int is_dead;

    const int matrix = blockIdx.x;
    const cuDoubleComplex* source = matrices + (size_t)matrix * size * size;

    for (int index = threadIdx.x; index < size * size; index += blockDim.x) {
        work[index] = source[index];
    }
    if (threadIdx.x == 0) {
        is_dead = 0;
    }
    __syncthreads();

    // relative tolerance from the largest magnitude of the matrix, like the
    // scalar reference which takes it once from the initial matrix
    double magnitude = 0.0;
    for (int index = threadIdx.x; index < size * size; index += blockDim.x) {
        magnitude = fmax(magnitude, hypot(work[index].x, work[index].y));
    }
    partial[threadIdx.x] = magnitude;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmax(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const double tolerance = relative_tolerance * fmax(partial[0], 1e-300);

    int sign = 1;
    cuDoubleComplex value = make_cuDoubleComplex(1.0, 0.0);
    for (int k = 0; k + 1 < size; k += 2) {
        if (threadIdx.x == 0) {
            candidate_of_pivot = -1;
            const cuDoubleComplex pivot = work[k * size + k + 1];
            if (hypot(pivot.x, pivot.y) < tolerance) {
                for (int j = k + 2; j < size; ++j) {
                    const cuDoubleComplex entry = work[k * size + j];
                    if (hypot(entry.x, entry.y) > tolerance) {
                        candidate_of_pivot = j;
                        break;
                    }
                }
                if (candidate_of_pivot < 0) {
                    is_dead = 1;   // no pivot and no candidate: this Pfaffian is 0
                }
            }
        }
        __syncthreads();
        const int candidate = candidate_of_pivot;
        if (candidate >= 0) {
            // the transposition of the two rows and columns, A -> P A P^T
            for (int index = threadIdx.x; index < size; index += blockDim.x) {
                const cuDoubleComplex swap = work[index * size + k + 1];
                work[index * size + k + 1] = work[index * size + candidate];
                work[index * size + candidate] = swap;
            }
            __syncthreads();
            for (int index = threadIdx.x; index < size; index += blockDim.x) {
                const cuDoubleComplex swap = work[(k + 1) * size + index];
                work[(k + 1) * size + index] = work[candidate * size + index];
                work[candidate * size + index] = swap;
            }
            if (threadIdx.x == 0) {
                sign = -sign;
            }
            __syncthreads();
        }
        const cuDoubleComplex pivot = work[k * size + k + 1];
        if (threadIdx.x == 0) {
            value = cmul(value, pivot);
        }
        const cuDoubleComplex inverse = cdiv(make_cuDoubleComplex(1.0, 0.0), pivot);
        // each thread owns whole rows of the trailing block, so every pair
        // (i, j) with i < j is written by exactly one thread
        for (int i = k + 2 + threadIdx.x; i < size; i += blockDim.x) {
            const cuDoubleComplex upper_left = work[k * size + i];
            const cuDoubleComplex lower_left = work[(k + 1) * size + i];
            for (int j = i + 1; j < size; ++j) {
                const cuDoubleComplex difference =
                    csub(cmul(upper_left, work[(k + 1) * size + j]),
                         cmul(work[k * size + j], lower_left));
                const cuDoubleComplex entry =
                    csub(work[i * size + j], cmul(difference, inverse));
                work[i * size + j] = entry;
                work[j * size + i] = make_cuDoubleComplex(-entry.x, -entry.y);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const cuDoubleComplex result = make_cuDoubleComplex(sign * value.x, sign * value.y);
        results[matrix] = is_dead ? make_cuDoubleComplex(0.0, 0.0) : result;
    }
}

// One thread per operator of the batch: it writes the whole mode-basis vector.
__global__ void operator_vectors_kernel(const cuDoubleComplex* __restrict__ coeff,
                                        const int64_t* __restrict__ codes,
                                        cuDoubleComplex* __restrict__ vectors,
                                        int num_modes, int coeff_stride,
                                        int64_t rows) {
    const int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int kind = (int)codes[row * 2];
    const int index = (int)codes[row * 2 + 1];
    cuDoubleComplex* target = vectors + row * (2 * num_modes);
    for (int k = 0; k < 2 * num_modes; ++k) {
        target[k] = make_cuDoubleComplex(0.0, 0.0);
    }
    if (kind == 0) {
        target[index] = make_cuDoubleComplex(1.0, 0.0);
    } else if (kind == 1) {
        target[num_modes + index] = make_cuDoubleComplex(1.0, 0.0);
    } else {
        // gamma_mu = sum_k (2 conj(coeff[k, mu]) d_k + 2 coeff[k, mu] d_k^dag)
        for (int k = 0; k < num_modes; ++k) {
            const cuDoubleComplex entry = coeff[k * coeff_stride + index];
            target[k] = make_cuDoubleComplex(2.0 * entry.x, -2.0 * entry.y);
            target[num_modes + k] = make_cuDoubleComplex(2.0 * entry.x, 2.0 * entry.y);
        }
    }
}

// One block per matrix: the upper triangle of <0|A_i A_j|0>, antisymmetrised.
__global__ void vacuum_contraction_kernel(const cuDoubleComplex* __restrict__ vectors,
                                          cuDoubleComplex* __restrict__ matrices,
                                          int num_modes, int length) {
    const int matrix = blockIdx.x;
    const cuDoubleComplex* source = vectors + (size_t)matrix * length * 2 * num_modes;
    cuDoubleComplex* target = matrices + (size_t)matrix * length * length;
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        target[i * length + i] = make_cuDoubleComplex(0.0, 0.0);
        for (int j = i + 1; j < length; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
            for (int k = 0; k < num_modes; ++k) {
                sum = cadd(sum, cmul(source[i * 2 * num_modes + k],
                                     source[j * 2 * num_modes + num_modes + k]));
            }
            target[i * length + j] = sum;
            target[j * length + i] = make_cuDoubleComplex(-sum.x, -sum.y);
        }
    }
}

}  // namespace

torch::Tensor pfaffian(torch::Tensor matrices) {
    TORCH_CHECK(matrices.is_cuda(), "pfaffian expects a CUDA tensor");
    TORCH_CHECK(matrices.scalar_type() == torch::kComplexDouble
                    || matrices.scalar_type() == torch::kFloat64,
                "pfaffian expects a float64 or complex128 tensor");
    TORCH_CHECK(matrices.dim() >= 3, "pfaffian expects (..., m, m) with a batch dimension");
    TORCH_CHECK(matrices.size(-2) == matrices.size(-1),
                "pfaffian expects square matrices");
    const c10::cuda::CUDAGuard device_guard(matrices.device());
    const bool real_input = matrices.scalar_type() == torch::kFloat64;
    auto source = real_input ? matrices.to(torch::kComplexDouble).contiguous()
                             : matrices.contiguous();
    const int size = (int)source.size(-1);
    std::vector<int64_t> batch(source.sizes().begin(), source.sizes().end() - 2);
    int64_t count = 1;
    for (int64_t extent : batch) {
        count *= extent;
    }
    auto results = torch::empty(batch, source.options());
    if (count == 0) {
        return real_input ? at::real(results) : results;
    }
    if (size % 2 != 0) {
        results.zero_();
    } else if (size == 0) {
        results.fill_(1.0);
    } else {
        TORCH_CHECK(size <= MAX_SIZE, "pfaffian supports matrices up to ", MAX_SIZE);
        pfaffian_kernel<<<(unsigned)count, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const cuDoubleComplex*>(source.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cuDoubleComplex*>(results.data_ptr<c10::complex<double>>()),
            size, 1e-12);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return real_input ? at::real(results) : results;
}

torch::Tensor operator_vectors(torch::Tensor coeff, torch::Tensor codes) {
    TORCH_CHECK(coeff.is_cuda() && codes.is_cuda(), "operator_vectors expects CUDA tensors");
    TORCH_CHECK(coeff.device() == codes.device(),
                "operator_vectors expects coeff and codes on the same CUDA device");
    TORCH_CHECK(coeff.scalar_type() == torch::kComplexDouble,
                "operator_vectors expects a complex128 coeff");
    TORCH_CHECK(codes.scalar_type() == torch::kInt64, "operator_vectors expects int64 codes");
    TORCH_CHECK(coeff.dim() == 2 && coeff.size(1) == 2 * coeff.size(0),
                "operator_vectors expects coeff of shape (n, 2n)");
    TORCH_CHECK(codes.dim() >= 2 && codes.size(-1) == 2,
                "operator_vectors expects codes of shape (..., m, 2)");
    const c10::cuda::CUDAGuard device_guard(coeff.device());
    auto coefficient = coeff.contiguous();
    auto code = codes.contiguous();
    const int num_modes = (int)coefficient.size(0);
    const int coeff_stride = (int)coefficient.size(1);
    const int64_t rows = code.numel() / 2;
    std::vector<int64_t> shape(code.sizes().begin(), code.sizes().end() - 1);
    shape.push_back(2 * num_modes);
    auto vectors = torch::empty(shape, coefficient.options());
    if (rows == 0) {
        return vectors;
    }
    const int64_t blocks = (rows + BLOCK - 1) / BLOCK;
    operator_vectors_kernel<<<(unsigned)blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cuDoubleComplex*>(coefficient.data_ptr<c10::complex<double>>()),
        code.data_ptr<int64_t>(),
        reinterpret_cast<cuDoubleComplex*>(vectors.data_ptr<c10::complex<double>>()),
        num_modes, coeff_stride, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return vectors;
}

torch::Tensor vacuum_contraction(torch::Tensor vectors, int64_t num_modes) {
    TORCH_CHECK(vectors.is_cuda(), "vacuum_contraction expects a CUDA tensor");
    TORCH_CHECK(vectors.scalar_type() == torch::kComplexDouble,
                "vacuum_contraction expects a complex128 tensor");
    TORCH_CHECK(vectors.dim() >= 3, "vacuum_contraction expects (..., m, 2n)");
    TORCH_CHECK(vectors.size(-1) == 2 * num_modes,
                "vacuum_contraction expects vectors of length 2 * num_modes");
    const c10::cuda::CUDAGuard device_guard(vectors.device());
    auto source = vectors.contiguous();
    const int length = (int)source.size(-2);
    std::vector<int64_t> batch(source.sizes().begin(), source.sizes().end() - 2);
    int64_t count = 1;
    for (int64_t extent : batch) {
        count *= extent;
    }
    std::vector<int64_t> shape(source.sizes().begin(), source.sizes().end() - 2);
    shape.push_back(length);
    shape.push_back(length);
    auto matrices = torch::empty(shape, source.options());
    if (count == 0 || length == 0) {
        return matrices.zero_();
    }
    vacuum_contraction_kernel<<<(unsigned)count, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cuDoubleComplex*>(source.data_ptr<c10::complex<double>>()),
        reinterpret_cast<cuDoubleComplex*>(matrices.data_ptr<c10::complex<double>>()),
        (int)num_modes, length);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return matrices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pfaffian", &pfaffian, "batched Pfaffian of antisymmetric matrices");
    m.def("operator_vectors", &operator_vectors, "batched mode-basis vectors of operator codes");
    m.def("vacuum_contraction", &vacuum_contraction, "batched vacuum contraction matrices");
}
