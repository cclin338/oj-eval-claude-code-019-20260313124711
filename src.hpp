#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K matrix in HBM (more efficient for Concat)
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    gpu_sim.Copy(keys[0], K, Position::kInGpuHbm);

    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_K_" + std::to_string(j));
      gpu_sim.Concat(K, keys[j], temp, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(K);
      K = temp;
    }

    // Move K to SRAM and transpose
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.Transpose(K, Position::kInSharedMemory);

    // Compute Q × K^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K, QK);
    gpu_sim.ReleaseMatrix(K);

    // Apply exp for softmax
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp_" + std::to_string(i));
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // Compute softmax row-wise - optimized to avoid extra Concat
    // Pre-allocate the full softmax matrix
    Matrix* softmax_QK = nullptr;

    for (size_t row_idx = 0; row_idx <= i; ++row_idx) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(row_idx));
      gpu_sim.GetRow(QK_exp, row_idx, row, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row_idx));
      gpu_sim.Sum(row, row_sum);

      Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(row_idx));
      gpu_sim.MatDiv(row, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (row_idx == 0) {
        softmax_QK = normalized_row;
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("temp_softmax_" + std::to_string(row_idx));
        gpu_sim.Concat(softmax_QK, normalized_row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        gpu_sim.ReleaseMatrix(normalized_row);
        softmax_QK = temp;
      }
    }
    gpu_sim.ReleaseMatrix(QK_exp);

    // Build V matrix in HBM
    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
    gpu_sim.Copy(values[0], V, Position::kInGpuHbm);

    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_V_" + std::to_string(j));
      gpu_sim.Concat(V, values[j], temp, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V);
      V = temp;
    }

    // Move V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(V);

    // Compute softmax(QK) × V
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_QK, V, result);
    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
