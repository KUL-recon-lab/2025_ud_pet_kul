#!/usr/bin/env bash
# Convert a nested DICOM tree to NIfTI.
# - Detects compressed transfer syntaxes per series (folder with .dcm files)
# - Decompresses only when needed (prefers gdcmconv -w; falls back to dcmdjpeg +te)
# - Runs dcm2niix on either original or decompressed series
# - PRESERVES nested directory structure under OUT_ROOT
# - Copies one representative DICOM from each series to the output dir as "_sample.dcm"
#
# Usage:
#   ./convert_all.sh [-n] [-o OUT_DIR] [-v] [ROOT]
#
# Options:
#   -o OUT_DIR   Output root directory (default: nifti_out)
#   -n           Cold mode (dry-run): print actions, do not execute commands
#   -v           Verbose: show success lines from dcm2niix and file copy
#   -h           Show this help and exit
#
# Requirements:
#   - dcm2niix (always)
#   - At least one of: gdcmconv (recommended) OR dcmdjpeg
#
set -uo pipefail   # note: no '-e' so one failure won't abort the whole batch

# ---------- defaults ----------
OUT_ROOT="nifti_out"
DRYRUN=0
VERBOSE=0
ROOT="."

print_help() {
cat <<'EOF'
Convert a nested DICOM tree to NIfTI (preserving folder structure).

Usage:
  ./convert_all.sh [-n] [-o OUT_DIR] [-v] [ROOT]

Options:
  -o OUT_DIR   Output root directory (default: nifti_out)
  -n           Cold mode (dry-run): print actions, do not execute commands
  -v           Verbose: show success lines from dcm2niix and sample DICOM copy
  -h           Show this help and exit

Notes:
  - Detects per-series transfer syntax; decompresses only when needed.
  - Decompression prefers 'gdcmconv -w'; falls back to 'dcmdjpeg +te'.
  - Output mirrors the input folder structure under OUT_DIR.
  - For each series, one original DICOM is copied to the output dir as "_sample.dcm".
  - In dry-run, nothing is created; only intended actions are printed.
EOF
}

# ---------- parse args ----------
while getopts ":o:nvh" opt; do
  case "$opt" in
    o) OUT_ROOT="$OPTARG" ;;
    n) DRYRUN=1 ;;
    v) VERBOSE=1 ;;
    h) print_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; print_help; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ $# -ge 1 ]]; then
  ROOT="$1"
fi

# ---------- helpers ----------
have_cmd() { command -v "$1" >/dev/null 2>&1; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
info() { printf '%s\n' "$*"; }

# Execute or just print (dry-run)
run() {
  if [[ $DRYRUN -eq 1 ]]; then
    printf '[DRY] %s\n' "$*" >&2
    return 0
  else
    eval "$@"
  fi
}

# Return 0 if directory contains any *.dcm, 1 otherwise
dir_has_dcm() {
  find "$1" -maxdepth 1 -type f -iname '*.dcm' -print -quit | grep -q .
}

# Determine transfer syntax UID of a sample file (robust).
get_transfer_syntax() {
  local f="$1" ts=""
  if have_cmd gdcminfo; then
    ts="$(gdcminfo "$f" 2>/dev/null | awk -F': ' '/Transfer Syntax/ {print $2; exit}' || true)"
  fi
  if [[ -z "$ts" ]] && have_cmd dcmdump; then
    ts="$( { dcmdump +M +P "0002,0010" "$f" 2>/dev/null || true; } \
           | sed -n 's/.*\[\(.*\)\].*/\1/p' | head -n1 )"
  fi
  printf '%s' "$ts"
}

# Return 0 if uncompressed TS, 1 otherwise
is_uncompressed_ts() {
  case "$1" in
    1.2.840.10008.1.2|1.2.840.10008.1.2.1|1.2.840.10008.1.2.2) return 0 ;;
    *) return 1 ;;
  esac
}

# Decompress one series directory into a temp dir; echo that temp dir path on success.
# If any file fails to decompress, we report and return non-zero without aborting the script.
decompress_series() {
  local src_dir="$1" tmpdir rc=0
  tmpdir="$(mktemp -d -t dcm_uncomp_XXXXXX)"

  if have_cmd gdcmconv; then
    while IFS= read -r -d '' f; do
      if [[ $DRYRUN -eq 1 ]]; then
        printf '[DRY] gdcmconv -w "%s" "%s/%s"\n' "$f" "$tmpdir" "$(basename "$f")"
      else
        if ! gdcmconv -w "$f" "$tmpdir/$(basename "$f")" 2>/dev/null; then
          warn "gdcmconv failed: $f"
          rc=1
          break
        fi
      fi
    done < <(find "$src_dir" -maxdepth 1 -type f -iname '*.dcm' -print0)
  else
    while IFS= read -r -d '' f; do
      if [[ $DRYRUN -eq 1 ]]; then
        printf '[DRY] dcmdjpeg +te "%s" "%s/%s"\n' "$f" "$tmpdir" "$(basename "$f")"
      else
        if ! dcmdjpeg +te "$f" "$tmpdir/$(basename "$f")" 2>/dev/null; then
          warn "dcmdjpeg failed: $f"
          rc=1
          break
        fi
      fi
    done < <(find "$src_dir" -maxdepth 1 -type f -iname '*.dcm' -print0)
  fi

  if [[ $rc -ne 0 ]]; then
    [[ $DRYRUN -eq 1 ]] || rm -rf "$tmpdir"
    return 1
  fi

  echo "$tmpdir"
  return 0
}

# Compute OUTDIR preserving nested structure under OUT_ROOT
compute_outdir() {
  local dir_abs="$1" root_abs="$2" rel=""
  if have_cmd realpath; then
    rel="$(realpath --relative-to="$root_abs" "$dir_abs" 2>/dev/null || true)"
  fi
  if [[ -z "$rel" ]]; then
    case "$dir_abs" in
      "$root_abs"/*) rel="${dir_abs#"$root_abs"/}" ;;
      *) rel="$(basename "$dir_abs")" ;;
    esac
  fi
  printf '%s/%s' "$OUT_ROOT" "$rel"
}

# Copy one representative original DICOM into OUTDIR as _sample.dcm (no overwrite)
copy_sample_dicom() {
  local outdir="$1" sample="$2"
  local dst="$outdir/_sample.dcm"
  if [[ $DRYRUN -eq 1 ]]; then
    printf '[DRY] cp -n "%s" "%s"\n' "$sample" "$dst"
  else
    cp -n -- "$sample" "$dst" 2>/dev/null || true
    [[ $VERBOSE -eq 1 ]] && echo "  -> Copied sample DICOM: $dst"
  fi
}

# Clean up temp dirs on ctrl-c
TMP_DIRS=()
cleanup() {
  if [[ $DRYRUN -eq 0 ]]; then
    for d in "${TMP_DIRS[@]:-}"; do
      [[ -n "$d" && -d "$d" ]] && rm -rf "$d"
    done
  fi
}
trap cleanup EXIT INT TERM

# ---------- normalize ROOT to absolute ----------
ROOT_ABS="$(cd "$ROOT" && pwd)"
info "Root: $ROOT_ABS"
info "Output root: $OUT_ROOT"
[[ $DRYRUN -eq 1 ]] && info "Mode: DRY-RUN (no files will be written)"

# ---------- sanity checks ----------
if ! have_cmd dcm2niix; then
  if [[ $DRYRUN -eq 1 ]]; then warn "dcm2niix not found (would be required for a real run)."
  else echo "ERROR: dcm2niix not found in PATH." >&2; exit 1; fi
fi
if ! have_cmd gdcmconv && ! have_cmd dcmdjpeg; then
  if [[ $DRYRUN -eq 1 ]]; then warn "Neither gdcmconv nor dcmdjpeg found (would be required for compressed series)."
  else echo "ERROR: Need either 'gdcmconv' (recommended) or 'dcmdjpeg' for decompression." >&2; exit 1; fi
fi

# ---------- collect series dirs (absolute) ----------
mapfile -d '' SERIES_DIRS < <(find "$ROOT_ABS" -type f -iname '*.dcm' -printf '%h\0' | sort -zu)
TOTAL=${#SERIES_DIRS[@]}
if [[ $TOTAL -eq 0 ]]; then info "No DICOM files (*.dcm) found under '$ROOT_ABS'."; exit 0; fi
info "Found $TOTAL series directories."

# Create output root unless dry-run
if [[ $DRYRUN -eq 0 ]]; then run "mkdir -p \"$OUT_ROOT\""; else printf '[DRY] mkdir -p \"%s\"\n' "$OUT_ROOT"; fi

# ---------- main loop ----------
idx=0
for DIR in "${SERIES_DIRS[@]}"; do
  idx=$((idx+1))
  [[ -d "$DIR" ]] || continue
  dir_has_dcm "$DIR" || continue

  SAMPLE="$(find "$DIR" -maxdepth 1 -type f -iname '*.dcm' -print -quit)"
  [[ -n "$SAMPLE" ]] || continue

  TS="$(get_transfer_syntax "$SAMPLE")"
  OUTDIR="$(compute_outdir "$DIR" "$ROOT_ABS")"

  echo "---- [$idx/$TOTAL]"
  echo "Series: $DIR"
  echo "Sample: $SAMPLE"
  echo "Transfer Syntax: ${TS:-unknown}"
  echo "Output: $OUTDIR"

  USE_DIR="$DIR"
  TMP_DIR=""

  if [[ -z "$TS" ]] || ! is_uncompressed_ts "$TS"; then
    echo "  -> Compressed or unknown TS detected; attempting decompression…"
    if [[ $DRYRUN -eq 1 ]]; then
      echo "  [DRY] Would decompress series into a temp directory, then run dcm2niix from there."
      USE_DIR="/tmp/dcm_uncomp_SIMULATED"
    else
      if TMP_DIR="$(decompress_series "$DIR")"; then
        USE_DIR="$TMP_DIR"
        TMP_DIRS+=("$TMP_DIR")
        echo "  -> Decompressed to: $TMP_DIR"
      else
        warn "Decompression failed; will try dcm2niix on original (may fail)."
        USE_DIR="$DIR"
      fi
    fi
  else
    echo "  -> Uncompressed syntax; using originals."
  fi

  # Ensure OUTDIR exists
  if [[ $DRYRUN -eq 0 ]]; then run "mkdir -p \"$OUTDIR\""; else printf '[DRY] mkdir -p \"%s\"\n' "$OUTDIR"; fi

  # Run dcm2niix
  DCM2NIIX_CMD="dcm2niix -z y -b y -o \"$OUTDIR\" -f \"%3s_%p_%t_%s\" \"$USE_DIR\""
  if [[ $DRYRUN -eq 1 ]]; then
    printf '[DRY] %s\n' "$DCM2NIIX_CMD"
  else
    if ! run "$DCM2NIIX_CMD"; then
      warn "dcm2niix FAILED for $DIR"
    elif [[ $VERBOSE -eq 1 ]]; then
      echo "  -> NIfTI written to: $OUTDIR"
    fi
  fi

  # Copy one representative *original* DICOM into OUTDIR
  copy_sample_dicom "$OUTDIR" "$SAMPLE"

  # Cleanup per-series temp dir immediately to save /tmp space
  if [[ $DRYRUN -eq 0 && -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    run "rm -rf \"$TMP_DIR\""
    # also remove from trap list
    for i in "${!TMP_DIRS[@]}"; do
      [[ "${TMP_DIRS[$i]}" == "$TMP_DIR" ]] && unset 'TMP_DIRS[i]'
    done
  fi
done

echo "All done. Outputs under: $OUT_ROOT"

