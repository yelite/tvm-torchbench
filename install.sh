set -euxo pipefail

TVM_GIT_REPO=$1 # e.g. https://github.com/apache/tvm
TVM_GIT_HASH=$2 # e.g. ada4c46f095f876efd97c4d0a3bf8860d7c5d5e8
TVM_DIR=tvm

cuda_load() {
  export CUDA_HOME="/usr/local/cuda"
  if [ -d $CUDA_HOME ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-""}
    echo "CUDA_HOME = $CUDA_HOME"
  else
    echo "Not exist: $CUDA_HOME"
    unset CUDA_HOME
  fi
}

setup_conda_env() { 
  export CONDA_HOME=$HOME/miniconda
  export PATH="$CONDA_HOME/bin:$PATH"
  __conda_setup="$($CONDA_HOME/bin/conda 'shell.bash' 'hook' 2> /dev/null)"
  unset __conda_setup

  conda env create -f=./environment.yml -n python-tvm-torchbench
  conda activate python-tvm-torchbench
}

clone_and_compile_tvm() {
  rm -rf $TVM_DIR
  git clone "$TVM_GIT_REPO" --recursive $TVM_DIR

  pushd $TVM_DIR

  git checkout $TVM_GIT_HASH
  mkdir build
  cp cmake/config.cmake build/config.cmake
  echo "set(CMAKE_BUILD_TYPE Release)"         >> build/config.cmake
  echo "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)" >> build/config.cmake
  echo "set(USE_LLVM ON)"                      >> build/config.cmake
  echo "set(USE_CUDA ON)"                      >> build/config.cmake
  echo "set(USE_CURAND ON)"                    >> build/config.cmake
  echo "set(USE_PT_TVMDSOOP ON)"               >> build/config.cmake

  cd build && cmake .. && make -j$(nproc)

  popd
}

cuda_load
setup_conda_env
clone_and_compile_tvm
