#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="${1:-/home/jwheojjang/scratch}"
UUID="d19756ba-3580-4830-baad-2f6e309fd80f"

if findmnt -rno TARGET "UUID=${UUID}" >/dev/null 2>&1; then
  CURRENT_TARGET="$(findmnt -rno TARGET "UUID=${UUID}")"
  echo "DOSCRATCH is already mounted at ${CURRENT_TARGET}"
  if ! touch "${CURRENT_TARGET}/.write_test" >/dev/null 2>&1; then
    echo "Mounted filesystem is not writable; remounting read-write."
    sudo mount -o remount,rw,noatime "${CURRENT_TARGET}"
  fi
  rm -f "${CURRENT_TARGET}/.write_test"
  MOUNTPOINT="${CURRENT_TARGET}"
else
  sudo mkdir -p "${MOUNTPOINT}"
  sudo mount -o rw,noatime "UUID=${UUID}" "${MOUNTPOINT}"
  sudo chown "${USER}:${USER}" "${MOUNTPOINT}"
fi

mkdir -p \
  "${MOUNTPOINT}/sr-diffusion/data" \
  "${MOUNTPOINT}/sr-diffusion/datasets" \
  "${MOUNTPOINT}/sr-diffusion/checkpoints" \
  "${MOUNTPOINT}/sr-diffusion/cache" \
  "${MOUNTPOINT}/sr-diffusion/runs"

echo "Mounted DOSCRATCH at ${MOUNTPOINT}"
df -hT "${MOUNTPOINT}"
