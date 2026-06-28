#!/usr/bin/env bash
# System-level dependencies for Isaac Lab / Isaac Sim 5.1 install.
# Run once with sudo. Does NOT touch the NVIDIA driver, kernel modules, or
# GPU state — purely user-space packages, so it will not interfere with
# any running CUDA processes.
#
# Source for the required packages:
#   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
#   (cmake + build-essential are listed as Linux requirements)
# vulkan-tools is added so we have `vulkaninfo` for diagnostics.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Re-running under sudo..."
  exec sudo -E bash "$0" "$@"
fi

echo "==> apt-get update"
apt-get update

echo "==> apt-get install cmake build-essential vulkan-tools"
apt-get install -y cmake build-essential vulkan-tools

echo
echo "==> Versions installed:"
cmake --version | head -1
gcc --version | head -1
vulkaninfo --summary 2>/dev/null | grep -E "(GPU|driverName|apiVersion)" | head -10 || true

echo
echo "Done. No GPU/driver state was touched; running CUDA processes are unaffected."
