#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sudo bash cache_drop.sh --yes [iterations] [interval_seconds]

Drops the Linux page cache by writing 1 to /proc/sys/vm/drop_caches.
This is a system-wide operation. The defaults perform one drop with no delay.
To repeat the operation at a fixed interval:

  sudo bash cache_drop.sh --yes 100000 0.1
EOF
}

if [[ ${1:-} != "--yes" ]]; then
    usage >&2
    exit 2
fi

if [[ ${EUID} -ne 0 ]]; then
    echo "error: this operation requires root (run with sudo)" >&2
    exit 1
fi

iterations=${2:-1}
interval_seconds=${3:-0}

if [[ ! ${iterations} =~ ^[1-9][0-9]*$ ]]; then
    echo "error: iterations must be a positive integer" >&2
    exit 2
fi
if [[ ! ${interval_seconds} =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "error: interval_seconds must be a non-negative number" >&2
    exit 2
fi

echo "warning: dropping the Linux page cache system-wide ${iterations} time(s)" >&2
sync
for ((i = 0; i < iterations; i++)); do
    echo 1 > /proc/sys/vm/drop_caches
    if ((i + 1 < iterations)); then
        sleep "${interval_seconds}"
    fi
done
