#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // In round i (0-indexed in code), Q has shape [i+1, 512]
    // We use keys[0] to keys[i] and values[0] to values[i]

    // Move query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Concatenate all keys K[0]...K[i] to form K matrix [i+1, 512]
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    gpu_sim.Copy(keys[0], K, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_K_" + std::to_string(j));
      gpu_sim.Concat(K, keys[j], temp, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(K);
      K = temp;
    }

    // Move K to SRAM and transpose it
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.Transpose(K, Position::kInSharedMemory);

    // Compute Q × K^T, result shape: [i+1, i+1]
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K, QK);
    gpu_sim.ReleaseMatrix(K);

    // Apply Softmax row-wise
    // For each row, compute: softmax[j] = exp(QK[j]) / sum(exp(QK[row]))
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp_" + std::to_string(i));
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);

    // Process each row separately for softmax
    // QK_exp has shape [i+1, i+1]
    Matrix* softmax_QK = matrix_memory_allocator.Allocate("softmax_QK_" + std::to_string(i));

    // Extract each row, compute its sum, divide, and put back
    for (size_t row_idx = 0; row_idx <= i; ++row_idx) {
      Matrix* row = matrix_memory_allocator.Allocate("row_" + std::to_string(row_idx));
      gpu_sim.GetRow(QK_exp, row_idx, row, Position::kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row_idx));
      gpu_sim.Sum(row, row_sum);

      Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(row_idx));
      gpu_sim.MatDiv(row, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);

      // Concatenate this row to build the result
      if (row_idx == 0) {
        gpu_sim.Copy(normalized_row, softmax_QK, Position::kInSharedMemory);
      } else {
        Matrix* temp = matrix_memory_allocator.Allocate("temp_softmax_" + std::to_string(row_idx));
        gpu_sim.Concat(softmax_QK, normalized_row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        softmax_QK = temp;
      }
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    gpu_sim.ReleaseMatrix(QK_exp);

    // Concatenate all values V[0]...V[i] to form V matrix [i+1, 512]
    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));
    gpu_sim.Copy(values[0], V, Position::kInGpuHbm);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp = matrix_memory_allocator.Allocate("temp_V_" + std::to_string(j));
      gpu_sim.Concat(V, values[j], temp, 0, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(V);
      V = temp;
    }

    // Move V to SRAM
    gpu_sim.MoveMatrixToSharedMem(V);

    // Compute softmax(QK) × V, result shape: [i+1, 512]
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_QK, V, result);
    gpu_sim.ReleaseMatrix(softmax_QK);
    gpu_sim.ReleaseMatrix(V);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu