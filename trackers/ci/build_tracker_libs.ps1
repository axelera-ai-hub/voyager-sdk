# Copyright Axelera AI, 2026
# Build the tracker C++ shared libraries for Windows.
# Called by cibuildwheel's before-all hook (runs once, not per Python version).
#
# The Linux counterpart (build_tracker_deps.sh) also builds Eigen3 and OpenCV
# from source, because the manylinux images ship neither at a usable version.
# On Windows both come from vcpkg, installed before cibuildwheel runs, so this
# script only has to build the trackers themselves.
#
# axelera_runtime2/CMakeLists.txt consumes the result: import libraries from
# trackers/lib and DLLs from trackers/bin.
$ErrorActionPreference = 'Stop'

$TrackersDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $TrackersDir 'build-win'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'cmake was not found on PATH.'
}

$Toolchain = $env:CMAKE_TOOLCHAIN_FILE
if (-not $Toolchain) {
    if (-not $env:VCPKG_ROOT) {
        throw 'Set CMAKE_TOOLCHAIN_FILE or VCPKG_ROOT: the trackers need OpenCV and Eigen3 from vcpkg.'
    }
    $Toolchain = Join-Path $env:VCPKG_ROOT 'scripts\buildsystems\vcpkg.cmake'
}
if (-not (Test-Path $Toolchain)) {
    throw "vcpkg toolchain file not found: $Toolchain"
}
$Triplet = if ($env:VCPKG_TARGET_TRIPLET) { $env:VCPKG_TARGET_TRIPLET } else { 'x64-windows-release' }

Write-Host '=== Building tracker C++ libraries ==='
Write-Host "  Trackers dir: $TrackersDir"
Write-Host "  Toolchain:    $Toolchain"
Write-Host "  Triplet:      $Triplet"

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

# Visual Studio generators select their target architecture with -A, which the
# single-config generators reject, so only pass it when one is in use.
$PlatformArgs = @()
if (-not $env:CMAKE_GENERATOR -or $env:CMAKE_GENERATOR -like 'Visual Studio*') {
    $PlatformArgs = @('-A', 'x64')
}

# The example application needs highgui and dnn, which nothing in the wheel
# does, so it is skipped here as it is on Linux.
cmake -S $TrackersDir -B $BuildDir @PlatformArgs `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    "-DCMAKE_INSTALL_PREFIX=$TrackersDir" `
    -DCMAKE_BUILD_TYPE=Release `
    -DBUILD_TRACKER_EXAMPLES=OFF
if ($LASTEXITCODE -ne 0) { throw 'tracker configure failed' }

cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw 'tracker build failed' }

cmake --install $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw 'tracker install failed' }

# Fail here rather than leaving axelera_runtime2 to quietly skip the bindings:
# its tracker block is a soft gate on these files existing.
foreach ($lib in 'tracker_factory', 'axtracker', 'bytetrack', 'oc_sort', 'tracktrack') {
    foreach ($artifact in "lib\$lib.lib", "bin\$lib.dll") {
        $path = Join-Path $TrackersDir $artifact
        if (-not (Test-Path $path)) {
            throw "Missing tracker artifact: $path"
        }
    }
}

Write-Host '=== Tracker build complete ==='
Get-ChildItem (Join-Path $TrackersDir 'bin')
