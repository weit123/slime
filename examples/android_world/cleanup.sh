#!/bin/bash

# Clean up Android emulators, ADB servers, and cached AVDs created by slime.
#
# Usage:
#   bash examples/android_world/cleanup.sh              # Kill emulators + ADB only
#   bash examples/android_world/cleanup.sh --avd        # Also remove cloned AVD files
#   bash examples/android_world/cleanup.sh --all        # Remove AVDs + temp images + lock files
#
# The script reads config.yaml (next to this script) for paths and naming
# patterns so it only touches resources that belong to slime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

# ── Parse flags ──────────────────────────────────────────────────────────────
REMOVE_AVD=false
REMOVE_TEMP=false
for arg in "$@"; do
    case "$arg" in
        --avd)  REMOVE_AVD=true ;;
        --all)  REMOVE_AVD=true; REMOVE_TEMP=true ;;
        -h|--help)
            echo "Usage: $0 [--avd] [--all]"
            echo "  (none)  Kill emulators and ADB servers only"
            echo "  --avd   Also delete cloned AVD files (slime_aw_*)"
            echo "  --all   Also delete temp images and ADB lock files"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg (try --help)"
            exit 1
            ;;
    esac
done

# ── Read config ──────────────────────────────────────────────────────────────
yaml_value() {
    # Minimal YAML value reader (no dependencies). Works for simple key: value lines.
    local key="$1" file="$2"
    grep -E "^${key}:" "$file" 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | tr -d '"'
}

if [ -f "$CONFIG" ]; then
    AVD_NAME_PATTERN=$(yaml_value "base_avd_name_pattern" "$CONFIG")
    ANDROID_AVD_HOME=$(yaml_value "android_avd_home" "$CONFIG")
    ADB_PATH=$(yaml_value "adb_path" "$CONFIG")
    EMULATOR_PATH=$(yaml_value "emulator_path" "$CONFIG")
    NUM_WORKERS=$(yaml_value "num_workers" "$CONFIG")
    TEMP_PATH=$(yaml_value "temp_path" "$CONFIG")
else
    echo "Warning: config.yaml not found at $CONFIG, using defaults"
fi

AVD_NAME_PATTERN="${AVD_NAME_PATTERN:-slime_aw_{}}"
ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-/root/android/avd/}"
ADB_PATH="${ADB_PATH:-/root/android/platform-tools/adb}"
EMULATOR_PATH="${EMULATOR_PATH:-/root/android/emulator/emulator}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TEMP_PATH="${TEMP_PATH:-/tmp/android_world_images}"

# Derive the glob prefix from the pattern (e.g. "slime_aw_{}" -> "slime_aw_")
AVD_PREFIX="${AVD_NAME_PATTERN//\{\}/}"

echo "=========================================="
echo "slime Android World Cleanup"
echo "=========================================="
echo "  AVD home:     $ANDROID_AVD_HOME"
echo "  AVD pattern:  ${AVD_PREFIX}*"
echo "  ADB:          $ADB_PATH"
echo "  Workers:      $NUM_WORKERS"
echo ""

SCRIPT_PID=$$
killed=0
skipped=0
failed=0

# ── Helpers ──────────────────────────────────────────────────────────────────

kill_matching() {
    # Kill processes whose command line matches $1 (extended regex).
    # Skips this script, zombies, and grep/pgrep helpers.
    local pattern="$1"
    local label="${2:-$pattern}"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    [ -z "$pids" ] && { echo "  [ok] No $label processes"; return; }

    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        [ "$pid" = "$SCRIPT_PID" ] && continue

        local state cmd
        state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' \t\n\r' || echo "")
        [ -z "$state" ] && continue                           # already gone
        [[ "$state" == *Z* ]] && { skipped=$((skipped+1)); continue; }  # zombie

        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        [[ "$cmd" == *cleanup.sh* ]] && continue
        [[ "$cmd" == *pgrep* ]] && continue
        [[ "$cmd" == *grep* ]] && continue

        if kill -9 "$pid" 2>/dev/null; then
            echo "  killed PID $pid  ($cmd)"
            killed=$((killed+1))
        else
            failed=$((failed+1))
        fi
    done <<< "$pids"
}

# ── 1. Kill slime emulator processes ─────────────────────────────────────────
echo "[1/5] Killing emulator processes..."
# Target processes whose command line contains the slime AVD prefix
kill_matching "${AVD_PREFIX}" "slime emulator (${AVD_PREFIX}*)"
# Catch any generic emulator / qemu processes as well
kill_matching "qemu-system.*${AVD_PREFIX}" "qemu (${AVD_PREFIX}*)"

# ── 2. Kill per-worker ADB servers ──────────────────────────────────────────
echo ""
echo "[2/5] Killing ADB servers..."
# The pool starts one adb server per worker. Kill them via the binary.
if [ -x "$ADB_PATH" ]; then
    "$ADB_PATH" kill-server 2>/dev/null || true
fi
# Also kill any lingering adb processes
kill_matching "adb.*server" "adb server"
# Give ADB a moment to release ports
sleep 0.5

# ── 3. Remove cloned AVD files (optional) ───────────────────────────────────
echo ""
echo "[3/5] Cloned AVD cleanup..."
if $REMOVE_AVD; then
    avd_removed=0
    for i in $(seq 1 "$NUM_WORKERS"); do
        avd_name="${AVD_PREFIX}${i}"
        avd_dir="${ANDROID_AVD_HOME}${avd_name}.avd"
        avd_ini="${ANDROID_AVD_HOME}${avd_name}.ini"
        if [ -d "$avd_dir" ]; then
            rm -rf "$avd_dir" && echo "  removed $avd_dir" && avd_removed=$((avd_removed+1))
        fi
        if [ -f "$avd_ini" ]; then
            rm -f "$avd_ini" && echo "  removed $avd_ini"
        fi
    done
    [ "$avd_removed" -eq 0 ] && echo "  [ok] No cloned AVDs found"
else
    echo "  [skip] Pass --avd or --all to remove cloned AVD files"
fi

# ── 4. Remove temp image files (optional) ───────────────────────────────────
echo ""
echo "[4/5] Temp image cleanup..."
if $REMOVE_TEMP; then
    if [ -d "$TEMP_PATH" ]; then
        rm -rf "$TEMP_PATH"
        echo "  removed $TEMP_PATH"
    else
        echo "  [ok] No temp directory at $TEMP_PATH"
    fi

    # Also clean /tmp/android-* and /tmp/emulator-* scratch files
    tmp_removed=0
    for d in /tmp/android-* /tmp/emulator-*; do
        [ -e "$d" ] || continue
        rm -rf "$d" 2>/dev/null && tmp_removed=$((tmp_removed+1))
    done
    [ "$tmp_removed" -gt 0 ] && echo "  removed $tmp_removed /tmp scratch entries"

    # ADB lock files
    for lf in "$HOME/.android/adb_usb.ini.lock" "$HOME/.android/adbkey.lock"; do
        [ -f "$lf" ] && rm -f "$lf" 2>/dev/null && echo "  removed $lf"
    done
else
    echo "  [skip] Pass --all to remove temp files and ADB locks"
fi

# ── 5. Summary ──────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying..."
remaining=$(pgrep -f "${AVD_PREFIX}" 2>/dev/null | grep -v "^${SCRIPT_PID}$" || true)
if [ -n "$remaining" ]; then
    # Filter zombies and this script
    live=""
    while IFS= read -r pid; do
        [ -z "$pid" ] && continue
        state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' \t\n\r' || echo "")
        [[ -z "$state" || "$state" == *Z* ]] && continue
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        [[ "$cmd" == *cleanup.sh* ]] && continue
        live="$live $pid"
    done <<< "$remaining"
    if [ -n "$live" ]; then
        echo "  WARNING: still running:$live"
        for pid in $live; do
            ps -p "$pid" -o pid,state,args 2>/dev/null || true
        done
    else
        echo "  [ok] All slime emulator processes cleaned up"
    fi
else
    echo "  [ok] All slime emulator processes cleaned up"
fi

echo ""
echo "=========================================="
echo "Done. Killed $killed process(es), skipped $skipped zombie(s), $failed failure(s)."
echo "=========================================="
