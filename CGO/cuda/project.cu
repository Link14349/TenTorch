// CUDA implementation of the patch projection of CGO/gaussian.py: the projected
// Majorana string P_V gamma_A P_V on the mixed register of a patch, one thread per
// output element.
//
// Every label rotates to the canonical basis of the patch, gamma_a = sum_c R[c,
// position(a)] gamma'_c, and splits as gamma_a = M_a + Z P~_a: the mixed part acts on the
// mixed register, the pure part acts on the pure modes and carries Z = prod Z_mu over
// every mixed mode, because the Jordan-Wigner string of gamma_{2 mu} = Z^{<mu} X_mu runs
// over all modes below mu.  The pure modes are fixed to their ground state, so the pure
// part contracts by Wick's theorem and only the mixed part stays an operator.  With
// <0|P~_S|0> = Pf(A_S) the Pfaffian of the antisymmetrized two-point function of the
// pure parts,
//
//     A_ij = sum_{mu pure} ( R[2 mu, p_i] R[2 mu, p_j] + R[2 mu+1, p_i] R[2 mu+1, p_j]
//                          - i (-R[2 mu, p_i] R[2 mu+1, p_j] + R[2 mu+1, p_i] R[2 mu, p_j]) )
//
// for i < j and A_ji = -A_ij: the real part is the delta of gamma'_c^2 = 1 (it vanishes
// when every mode is pure, where the rotation is orthogonal), the imaginary part is the
// pure block of the canonical covariance, whose eigenvalue is 1 on every pure mode,
//
//     P_V gamma_A P_V = sum_{S even} (-1)^cross(S) <0|P~_S|0> prod_{i not in S} M_{a_i},
//
// where cross(S) counts the mixed labels that precede a selected pure label: each Z has
// to be moved past an M, and Z M = -M Z.  The 2^k subset weights and signs are computed
// once per term on the host from the rotation alone; the kernel contracts the mixed
// products elementwise and never builds the 2^p register of the pure modes.
//
// Build and load through CGO/cuda/project.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <complex>
#include <vector>

namespace {

constexpr int BLOCK = 256;      // threads per block, one thread per output element
constexpr int MAX_LABELS = 8;   // largest Majorana string of one term, 2^(k-1) subsets

using Complex = std::complex<double>;

// Pfaffian of a complex antisymmetric matrix by first-row expansion,
// Pf(A) = sum_j (-1)^(j+1) A[0, j] Pf(A without rows and columns 0 and j); an odd size
// never reaches the recursion.
Complex pfaffian(const Complex* matrix, int n) {
    if (n == 0) return Complex(1.0, 0.0);
    std::vector<Complex> minor((n - 2) * (n - 2));
    Complex total(0.0, 0.0);
    for (int column = 1; column < n; ++column) {
        for (int a = 0; a < n - 2; ++a) {
            const int row_a = a + 1 + (a + 1 >= column ? 1 : 0);
            for (int b = 0; b < n - 2; ++b) {
                const int row_b = b + 1 + (b + 1 >= column ? 1 : 0);
                minor[a * (n - 2) + b] = matrix[row_a * n + row_b];
            }
        }
        total += (column % 2 == 1 ? 1.0 : -1.0) * matrix[column] *
                 pfaffian(minor.data(), n - 2);
    }
    return total;
}

__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

// (row, row ^ 2^bit) entry of the mixed part M_a of the label at patch index `column`:
// gamma_{2 mu} = Z^{<mu} X_mu contributes 1, gamma_{2 mu+1} = Z^{<mu} Y_mu contributes
// +-i, and Z^{<mu} contributes the parity of the bits of the mixed modes before mu.
// The register bit of the mixed mode mu is `bit`, so the modes before it are the bits
// above `bit` and the string sign uses them.
__device__ __forceinline__ cuDoubleComplex mixed_entry(const double* __restrict__ rotation,
                                                       const int64_t dimension,
                                                       const int64_t column,
                                                       const int64_t* __restrict__ mixed,
                                                       const int64_t num_mixed,
                                                       const int64_t row,
                                                       const int64_t bit) {
    const int64_t mode = mixed[num_mixed - 1 - bit];
    const double string = (__popcll(static_cast<unsigned long long>(row >> (bit + 1))) % 2) ? -1.0 : 1.0;
    const double y_sign = ((row >> bit) & 1) ? 1.0 : -1.0;
    const double even = rotation[(2 * mode) * dimension + column];
    const double odd = rotation[(2 * mode + 1) * dimension + column];
    return make_cuDoubleComplex(string * even, string * odd * y_sign);
}

// The mixed product prod_i M_{a_i} at (row, column), one term of the subset sum.  Each
// M is a single Majorana, so every factor flips exactly one register bit: the product
// runs over the intermediate states, num_mixed choices per factor, and the last factor
// is the single-step test that only one bit may differ.  count == 0 is the identity.
__device__ cuDoubleComplex mixed_product(const int64_t* patch_indices,
                                         const int count,
                                         const int64_t row,
                                         const int64_t column,
                                         const double* __restrict__ rotation,
                                         const int64_t dimension,
                                         const int64_t* __restrict__ mixed,
                                         const int64_t num_mixed) {
    if (count == 0) return make_cuDoubleComplex(row == column ? 1.0 : 0.0, 0.0);
    if (count == 1) {
        const int64_t difference = row ^ column;
        if (difference == 0 || (difference & (difference - 1)) != 0) return make_cuDoubleComplex(0.0, 0.0);
        const int64_t bit = __ffsll(static_cast<long long>(difference)) - 1;
        return mixed_entry(rotation, dimension, patch_indices[0], mixed, num_mixed, row, bit);
    }
    cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
    for (int64_t bit = 0; bit < num_mixed; ++bit) {
        const int64_t next = row ^ (static_cast<int64_t>(1) << bit);
        const cuDoubleComplex step = mixed_entry(rotation, dimension, patch_indices[0], mixed, num_mixed, row, bit);
        if (step.x == 0.0 && step.y == 0.0) continue;
        total = cadd(total, cmul(step, mixed_product(patch_indices + 1, count - 1, next, column,
                                                     rotation, dimension, mixed, num_mixed)));
    }
    return total;
}

__global__ void project_kernel(const double* __restrict__ rotation,
                               const int64_t dimension,
                               const int64_t* __restrict__ mixed,
                               const int64_t num_mixed,
                               const int64_t* __restrict__ positions,
                               const int64_t* __restrict__ offsets,
                               const cuDoubleComplex* __restrict__ coefficients,
                               const cuDoubleComplex* __restrict__ weights,
                               const int64_t* __restrict__ masks,
                               const int64_t* __restrict__ weight_offsets,
                               const int64_t num_terms,
                               const int64_t elements,
                               const int64_t size,
                               cuDoubleComplex* __restrict__ out) {
    const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= elements) return;
    const int64_t row = element / size;
    const int64_t column = element % size;

    cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
    for (int64_t term = 0; term < num_terms; ++term) {
        const int64_t begin = offsets[term];
        const int labels = static_cast<int>(offsets[term + 1] - begin);
        cuDoubleComplex value = make_cuDoubleComplex(0.0, 0.0);
        for (int64_t subset = weight_offsets[term]; subset < weight_offsets[term + 1]; ++subset) {
            const int64_t mask = masks[subset];
            int64_t patch_indices[MAX_LABELS];
            int count = 0;
            for (int i = 0; i < labels; ++i)
                if (!((mask >> i) & 1)) patch_indices[count++] = positions[begin + i];
            value = cadd(value, cmul(weights[subset],
                                     mixed_product(patch_indices, count, row, column,
                                                   rotation, dimension, mixed, num_mixed)));
        }
        total = cadd(total, cmul(coefficients[term], value));
    }
    out[element] = total;
}

}  // namespace

torch::Tensor project(torch::Tensor rotation,
                      torch::Tensor mixed,
                      torch::Tensor pure,
                      torch::Tensor positions,
                      torch::Tensor offsets,
                      torch::Tensor coefficients) {
    TORCH_CHECK(rotation.is_cuda(), "rotation must be on CUDA");
    const c10::cuda::CUDAGuard device_guard(rotation.device());
    TORCH_CHECK(rotation.scalar_type() == torch::kFloat64, "rotation must be float64");
    TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == rotation.size(1),
                "rotation must be square, got ", rotation.sizes());
    TORCH_CHECK(coefficients.scalar_type() == torch::kComplexDouble, "coefficients must be complex128");
    const auto device = rotation.device();

    auto rotation_device = rotation.contiguous();
    auto host_rotation = rotation_device.to(torch::kCPU);
    auto host_mixed = mixed.contiguous().to(torch::kCPU).to(torch::kInt64);
    auto host_pure = pure.contiguous().to(torch::kCPU).to(torch::kInt64);
    auto host_positions = positions.contiguous().to(torch::kCPU).to(torch::kInt64);
    auto host_offsets = offsets.contiguous().to(torch::kCPU).to(torch::kInt64);
    const int64_t dimension = host_rotation.size(0);
    const int64_t num_mixed = host_mixed.numel();
    const int64_t num_pure = host_pure.numel();
    const int64_t num_terms = host_offsets.numel() - 1;
    TORCH_CHECK(host_offsets.numel() >= 1, "offsets must start with the first term");
    TORCH_CHECK(num_terms >= 0, "offsets must hold one boundary per term and one more");
    TORCH_CHECK(num_mixed + num_pure <= dimension / 2, "the modes of the patch exceed the rotation");
    TORCH_CHECK(num_mixed < 31, "the mixed register is too large, got ", num_mixed, " modes");

    const double* rotation_values = host_rotation.data_ptr<double>();
    const int64_t* mixed_values = host_mixed.data_ptr<int64_t>();
    const int64_t* pure_values = host_pure.data_ptr<int64_t>();
    const int64_t* position_values = host_positions.data_ptr<int64_t>();
    const int64_t* offset_values = host_offsets.data_ptr<int64_t>();
    for (int64_t term = 0; term < num_terms; ++term) {
        TORCH_CHECK(0 <= offset_values[term] && offset_values[term] <= offset_values[term + 1] &&
                    offset_values[term + 1] <= host_positions.numel(), "offsets must be increasing and inside positions");
    }

    // the 2^(k-1) even subsets of every term and their pure vacuum contraction
    std::vector<Complex> weights;
    std::vector<int64_t> masks;
    std::vector<int64_t> weight_offsets{0};
    for (int64_t term = 0; term < num_terms; ++term) {
        const int64_t begin = offset_values[term];
        const int64_t labels = offset_values[term + 1] - begin;
        TORCH_CHECK(labels <= MAX_LABELS, "a term with ", labels, " labels exceeds the ", MAX_LABELS,
                    " Majoranas of one term");
        std::vector<Complex> kappa(labels * labels, Complex(0.0, 0.0));
        for (int64_t i = 0; i < labels; ++i) {
            for (int64_t j = 0; j < labels; ++j) {
                const int64_t column_i = position_values[begin + i];
                const int64_t column_j = position_values[begin + j];
                TORCH_CHECK(0 <= column_i && column_i < dimension, "the patch index ", column_i, " is out of range");
                TORCH_CHECK(0 <= column_j && column_j < dimension, "the patch index ", column_j, " is out of range");
                double value = 0.0;     // the pure block of the canonical covariance
                double identity = 0.0;  // the delta of the two-point function, gamma'_c^2 = 1
                for (int64_t index = 0; index < num_pure; ++index) {
                    const int64_t mode = pure_values[index];
                    TORCH_CHECK(0 <= mode && mode < dimension / 2, "the pure mode ", mode, " is out of range");
                    const double even_i = rotation_values[(2 * mode) * dimension + column_i];
                    const double odd_i = rotation_values[(2 * mode + 1) * dimension + column_i];
                    const double even_j = rotation_values[(2 * mode) * dimension + column_j];
                    const double odd_j = rotation_values[(2 * mode + 1) * dimension + column_j];
                    value += -even_i * odd_j + odd_i * even_j;
                    identity += even_i * even_j + odd_i * odd_j;
                }
                // the two-point function of the pure parts, antisymmetrized for the Pfaffian
                kappa[i * labels + j] = (i == j) ? Complex(0.0, 0.0)
                                      : Complex((i < j ? 1.0 : -1.0) * identity, -value);
            }
        }
        for (int64_t mask = 0; mask < (static_cast<int64_t>(1) << labels); ++mask) {
            if (__builtin_popcountll(static_cast<unsigned long long>(mask)) % 2) continue;
            int crossings = 0;  // mixed labels before a selected pure label, each one a minus
            for (int64_t i = 0; i < labels; ++i)
                if ((mask >> i) & 1)
                    for (int64_t j = 0; j < i; ++j)
                        if (!((mask >> j) & 1)) ++crossings;
            std::vector<int> selected;
            for (int64_t i = 0; i < labels; ++i)
                if ((mask >> i) & 1) selected.push_back(static_cast<int>(i));
            std::vector<Complex> minor(selected.size() * selected.size());
            for (size_t a = 0; a < selected.size(); ++a)
                for (size_t b = 0; b < selected.size(); ++b)
                    minor[a * selected.size() + b] = kappa[selected[a] * labels + selected[b]];
            weights.push_back(pfaffian(minor.data(), static_cast<int>(selected.size())) *
                              (crossings % 2 ? -1.0 : 1.0));
            masks.push_back(mask);
        }
        weight_offsets.push_back(static_cast<int64_t>(weights.size()));
    }

    const auto int64_options = torch::TensorOptions().dtype(torch::kInt64);
    const auto complex_options = torch::TensorOptions().dtype(torch::kComplexDouble);
    auto weights_device = torch::from_blob(weights.data(), {static_cast<int64_t>(weights.size())}, complex_options).to(device);
    auto masks_device = torch::from_blob(masks.data(), {static_cast<int64_t>(masks.size())}, int64_options).to(device);
    auto weight_offsets_device = torch::from_blob(weight_offsets.data(), {static_cast<int64_t>(weight_offsets.size())}, int64_options).to(device);
    auto mixed_device = host_mixed.to(device);
    auto positions_device = host_positions.to(device);
    auto offsets_device = host_offsets.to(device);
    auto coefficients_device = coefficients.contiguous().to(device);

    const int64_t size = static_cast<int64_t>(1) << num_mixed;
    const int64_t elements = size * size;
    auto out = torch::empty({size, size}, complex_options.device(device));
    const int blocks = static_cast<int>((elements + BLOCK - 1) / BLOCK);
    project_kernel<<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        rotation_device.data_ptr<double>(), dimension, mixed_device.data_ptr<int64_t>(), num_mixed,
        positions_device.data_ptr<int64_t>(), offsets_device.data_ptr<int64_t>(),
        reinterpret_cast<const cuDoubleComplex*>(coefficients_device.data_ptr<c10::complex<double>>()),
        reinterpret_cast<const cuDoubleComplex*>(weights_device.data_ptr<c10::complex<double>>()),
        masks_device.data_ptr<int64_t>(), weight_offsets_device.data_ptr<int64_t>(),
        num_terms, elements, size, reinterpret_cast<cuDoubleComplex*>(out.data_ptr<c10::complex<double>>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project", &project, "projected Majorana strings on the mixed register of a patch");
}
