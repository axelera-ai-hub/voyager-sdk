#!/usr/bin/env bash
# Copyright Axelera AI, 2026
# Build tracker C++ libraries and their dependencies.
# Called by build_deps.sh after common dependencies are installed.
#
# This script:
#   1. Installs build tools (cmake, ninja, gcc-c++)
#   2. Installs Eigen3 headers (header-only, required by trackers)
#   3. Builds minimal OpenCV 4 from source (required by trackers)
#   4. Builds tracker C++ shared libraries
#
# The runtime2 CMakeLists.txt finds the pre-built tracker libs at
# ../trackers/lib/ and builds the _axtracker nanobind binding per Python version.
#
# Key design decisions:
#
# - OpenCV from source: manylinux_2_28 repos only have OpenCV 3.4.x via EPEL;
#   trackers need 4.x. We build only production modules (core, imgproc,
#   features2d, video, calib3d, flann). The track example app needs videoio/
#   highgui -- excluded via BUILD_TRACKER_EXAMPLES=OFF.
#
# - Shared libs + auditwheel: auditwheel vendors the .so files into the wheel
#   with hash-renamed SONAMEs, making it self-contained. This coexists safely
#   with opencv-python because Python loads extensions with RTLD_LOCAL (symbol
#   isolation) and no cv::Mat crosses the nanobind binding boundary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKERS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# /usr/local/share/pkgconfig is not in the default search path on EL8 manylinux.
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/local/share/pkgconfig:${PKG_CONFIG_PATH:-}"

OPENCV_VERSION="4.10.0"
EIGEN_VERSION="3.4.0"
NPROC="$(nproc)"

echo "=== Building tracker dependencies and libraries ==="
echo "  Trackers dir: $TRACKERS_DIR"
echo "  CPUs: $NPROC"

# --- 1. Install build tools ---
echo "--- Installing build tools ---"
# manylinux containers have gcc/g++ but may lack cmake/ninja.
# yum can fail due to GPG key issues in older manylinux images, so install
# cmake and ninja via pip (available in all manylinux containers).
yum install -y gcc-c++ 2>/dev/null || true
# Force-install a pinned cmake even if one is already on PATH: manylinux_2_28
# ships cmake 4.x via pipx, which removed compatibility with cmake_minimum_required(<3.5)
# and breaks OpenCV 4.10's OpenCVGenPkgconfig.cmake helper script.
/opt/python/cp310-cp310/bin/pip install --force-reinstall 'cmake<4'
ln -sf /opt/python/cp310-cp310/bin/cmake /usr/local/bin/cmake
export PATH="/opt/python/cp310-cp310/bin:$PATH"
hash -r
echo "  Using cmake: $(command -v cmake) ($(cmake --version | head -1))"
if ! command -v ninja &>/dev/null; then
    /opt/python/cp310-cp310/bin/pip install ninja
    ln -sf /opt/python/cp310-cp310/bin/ninja /usr/local/bin/ninja
fi

# --- 2. Install Eigen3 (header-only) ---
echo "--- Installing Eigen3 ---"
if pkg-config --exists eigen3 2>/dev/null; then
    echo "  Eigen3 already available ($(pkg-config --modversion eigen3))"
else
    echo "  Building Eigen3 ${EIGEN_VERSION} from source"
    curl -sL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" | tar xz -C /tmp
    cmake -S "/tmp/eigen-${EIGEN_VERSION}" -B /tmp/eigen-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_DOC=OFF
    ninja -C /tmp/eigen-build install
    rm -rf /tmp/eigen-${EIGEN_VERSION} /tmp/eigen-build
fi

# --- 3. Build minimal OpenCV 4 ---
# Production needs:
#   - Tracker libs: core, imgproc, features2d, video, calib3d, flann
#   - Runtime2 video decoder: core, imgproc, videoio
echo "--- Building OpenCV ${OPENCV_VERSION} (minimal) ---"
if pkg-config --exists opencv4 2>/dev/null; then
    echo "  OpenCV 4 already available ($(pkg-config --modversion opencv4))"
else
    curl -sL "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" | tar xz -C /tmp
    cmake -S "/tmp/opencv-${OPENCV_VERSION}" -B /tmp/opencv-build -G Ninja \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=core,imgproc,videoio,features2d,video,calib3d,flann \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python3=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GTK=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_V4L=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_LAPACK=OFF \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    ninja -C /tmp/opencv-build -j"$NPROC"
    ninja -C /tmp/opencv-build install
    PC_FILE=""
    for p in \
        /usr/local/lib/pkgconfig/opencv4.pc \
        /usr/local/lib64/pkgconfig/opencv4.pc \
        /usr/local/lib/pkgconfig/opencv.pc \
        /usr/local/lib64/pkgconfig/opencv.pc; do
        if [[ -f "$p" ]]; then
            PC_FILE="$p"
            break
        fi
    done
    if [[ -z "$PC_FILE" ]]; then
        GEN_PC="$(find /tmp/opencv-build -type f -name 'opencv*.pc' | head -1 || true)"
        if [[ -n "$GEN_PC" ]]; then
            mkdir -p /usr/local/lib/pkgconfig
            cp -a "$GEN_PC" /usr/local/lib/pkgconfig/opencv4.pc
            PC_FILE="/usr/local/lib/pkgconfig/opencv4.pc"
        else
            echo "ERROR: OpenCV pkg-config file was not generated" >&2
            exit 1
        fi
    fi
    echo "  Using OpenCV pkg-config file: $PC_FILE"
    ldconfig 2>/dev/null || true
    rm -rf "/tmp/opencv-${OPENCV_VERSION}" /tmp/opencv-build
fi

# --- 4. Build tracker C++ libraries ---
# Skip example app (needs highgui which is not in minimal OpenCV)
echo "--- Building tracker C++ libraries ---"
cd "$TRACKERS_DIR"
# Clean any stale CMake cache (e.g., from local development or previous runs)
rm -rf Release
mkdir -p Release
cmake -S . -B Release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$TRACKERS_DIR" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_TRACKER_EXAMPLES=OFF
ninja -C Release install

echo "=== Tracker build complete ==="
ls -la "$TRACKERS_DIR/lib/"
