Write-Host "========================================"
Write-Host " Setting up LLVM & MLIR on Windows"
Write-Host "========================================"

$LLVM_DIR = "D:\llvm-project"

if (-Not (Test-Path $LLVM_DIR)) {
    Write-Host "[1/2] Cloning LLVM project to $LLVM_DIR..."
    # Depth 1 and specific branch to save time and space
    git clone --depth 1 -b release/18.x https://github.com/llvm/llvm-project.git $LLVM_DIR
} else {
    Write-Host "[1/2] LLVM already cloned at $LLVM_DIR. Skipping clone."
}

Write-Host "[2/2] Configuring and Building LLVM with MLIR..."
$BUILD_DIR = "$LLVM_DIR\build"
if (-Not (Test-Path $BUILD_DIR)) {
    New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
}

Set-Location $BUILD_DIR

# Configure using Visual Studio generator
cmake -G "Visual Studio 17 2022" -A x64 ..\llvm `
   -Thost=x64 `
   -DLLVM_ENABLE_PROJECTS="mlir" `
   -DLLVM_TARGETS_TO_BUILD="host;ARM;AArch64" `
   -DCMAKE_BUILD_TYPE=Release `
   -DLLVM_ENABLE_ASSERTIONS=ON

# Build mlir
cmake --build . --target mlir --config Release

Write-Host "========================================"
Write-Host " LLVM & MLIR build successful!"
Write-Host " LLVM build directory: $BUILD_DIR"
Write-Host "========================================"
