ROOT=$(pwd)
CPU_CORES=$(nproc)  # Get number of CPU cores
PARALLEL_JOBS=$((CPU_CORES + 1))  # Optimal for make

cd /kaolin && pip install -e .
cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j${PARALLEL_JOBS}
