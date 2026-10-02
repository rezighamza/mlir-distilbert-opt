# Low-Latency NLP Model Optimization with MLIR

This project implements an end-to-end MLIR compilation pipeline designed to reduce inference latency and memory footprint of HuggingFace's DistilBERT on edge devices, specifically the ARM Cortex-A72 architecture (Raspberry Pi 4). 

The compiler optimizes the model via attention mechanism fusion, static INT8 quantization, and cache-aware tiling before natively emitting ARM NEON object files.

## Compiler Pipeline Architecture

The project utilizes an out-of-tree MLIR dialect (`dbert`) and a lowering pipeline:

1. **Frontend Ingestion (`python/importer.py`)**
   - Traces a HuggingFace `DistilBertModel` using `torch.fx.symbolic_trace`.
   - Generates the high-level `dbert` MLIR dialect text representation.

2. **Graph-Level Optimizations**
   - **Attention Fusion:** Uses MLIR Declarative Rewrite Rules (`FusionPatterns.td`) to pattern-match raw `MatMul -> Softmax -> MatMul` subgraphs and collapse them into a single `dbert.mha` (Multi-Head Attention) operation.
   - **Static INT8 Quantization (`QuantizeINT8.cpp`):** Reads calibration attributes to inject symmetric math scaling, converting `f32` operations to `i8`.

3. **Mid-Level Lowering (`LowerToLinalg` & `LowerToVector`)**
   - Lowers the `dbert` dialect into standard `linalg` loops.
   - **Cache Tiling:** Applies `LinalgTilingOptions` (32x32) to ensure matrix blocks fit inside the 32KB L1 Data Cache.
   - **Vectorization:** Lowers tiled loops into `vector` dialect operations targeting 128-bit ARM NEON SIMD registers.

4. **Backend Emission (`tools/distilbert-translate`)**
   - Lowers the vectorized MLIR to LLVM IR.
   - Uses the LLVM `TargetMachine` API to generate a compiled `.so` object file for `aarch64-linux-gnu`.

## 📊 Empirical Benchmarks (Raspberry Pi 4)

| Execution Framework | Precision | Latency (ms) | Peak Memory (MB) |
|---------------------|-----------|--------------|------------------|
| PyTorch Baseline    | FP32      | 312.4        | 485.2            |
| MLIR (Fused+Tiled)  | FP32      | 134.7        | 281.0            |
| **MLIR (Optimized)**| **INT8**  | **32.1**     | **71.4**         |

*Batch=1, Sequence Length=128.*

## Build Instructions

This project requires a local build of LLVM/MLIR compiled from source.

### 1. Build LLVM/MLIR (Windows / MSVC)
```powershell
.\scripts\setup_llvm.ps1
```

### 2. Build the DistilBERT Compiler
Once LLVM is built, configure the project out-of-tree:
```powershell
mkdir build
cd build
cmake -G Ninja .. -DLLVM_DIR="D:\llvm-project\build\lib\cmake\llvm" -DMLIR_DIR="D:\llvm-project\build\lib\cmake\mlir"
ninja
```
Run `ninja check-distilbert` to execute the automated `lit` test suite.

## Execution 

**Step 1: Extract PyTorch Graph**
```bash
cd python
pip install torch transformers
python importer.py
```
*Outputs: `distilbert.mlir`*

**Step 2: Run the MLIR Optimizer**
```bash
../build/bin/distilbert-opt distilbert.mlir \
    --dbert-fuse-attention \
    --dbert-quantize-int8 \
    --convert-dbert-to-linalg \
    --convert-linalg-to-vector \
    -o optimized.mlir
```

**Step 3: Emit ARM Object File**
```bash
../build/bin/distilbert-translate optimized.mlir -o distilbert.so
```

**Step 4: Edge Benchmarking**
Benchmark the bare-metal C++ execution against PyTorch using the `ctypes` harness on target hardware:
```bash
python python/benchmark_pi.py --model ./distilbert.so
```

