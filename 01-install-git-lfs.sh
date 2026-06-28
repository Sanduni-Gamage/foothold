#!/usr/bin/env bash
# Install git-lfs (required to pull Unitree's Go2 USD asset from
# unitree_model). Pure userspace package install — does not touch the
# NVIDIA driver or running CUDA processes.
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
  echo "Re-running under sudo..."
  exec sudo -E bash "$0" "$@"
fi
echo "==> apt-get install git-lfs"
apt-get update -qq
apt-get install -y git-lfs
echo "==> git lfs version: $(git-lfs version)"
echo "Done."
