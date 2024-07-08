# Correct
../llvm-project/build/bin/mlir-opt \
  --one-shot-bufferize \
  --convert-linalg-to-loops \
  --lower-affine \
  --buffer-deallocation-pipeline \
  --scf-for-loop-canonicalization \
  --convert-scf-to-cf \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --convert-math-to-llvm \
  --convert-index-to-llvm \
  --convert-vector-to-llvm \
  --reconcile-unrealized-casts \
  test/mlir/lowered_to_std.mlir

