#!/usr/bin/env bash
# One-time setup of a Linux machine / cloud environment for this project.
#
#   bash setup/cloud_setup.sh
#
# Environment variables:
#   SYNTH_WORK_ROOT     work folder for data, caches, outputs (default ~/synth_aerial_work)
#   Place licensed asset ZIPs in ../assets before running setup.
#   BLENDER_VERSION     default 5.2.2
#
# Installs Blender (Linux build) and the Python packages, downloads the Copernicus
# GLO-30 elevation tiles, fetches and ingests the assets, and builds the template.
# Without a GPU, Cycles renders on the CPU (much slower); planning, rule checks
# and tests do not need Blender at all.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${SYNTH_WORK_ROOT:-$HOME/synth_aerial_work}"
BV="${BLENDER_VERSION:-5.2.2}"
BMAJ="${BV%.*}"
mkdir -p "$WORK/data/dtm/raw" "$WORK/cache" "$WORK/outputs" "$WORK/templates" "$WORK/tools"

echo "== system libraries Blender needs headless"
if command -v apt-get >/dev/null; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo "sudo")
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq libxi6 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 xz-utils curl unzip python3-venv >/dev/null
fi

echo "== Blender $BV"
if [ ! -x "$WORK/tools/blender/blender" ]; then
  curl -fsSL "https://download.blender.org/release/Blender${BMAJ}/blender-${BV}-linux-x64.tar.xz" -o "$WORK/tools/blender.tar.xz"
  tar -xf "$WORK/tools/blender.tar.xz" -C "$WORK/tools"
  mv "$WORK/tools/blender-${BV}-linux-x64" "$WORK/tools/blender"
  rm "$WORK/tools/blender.tar.xz"
fi
export BLENDER="$WORK/tools/blender/blender"

echo "== Python environment"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/pip" install -q -r "$PROJECT/requirements.txt"

echo "== elevation data (Copernicus DEM GLO-30, AWS open data)"
for T in N32 N33; do
  OUT="$WORK/data/dtm/raw/Copernicus_${T}_E035.tif"
  [ -f "$OUT" ] || curl -fsSL "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_${T}_00_E035_00_DEM/Copernicus_DSM_COG_10_${T}_00_E035_00_DEM.tif" -o "$OUT"
done
if [ -f "$PROJECT/setup/aoi_korazim.gpkg" ]; then
  cp "$PROJECT/setup/aoi_korazim.gpkg" "$WORK/data/aoi_korazim.gpkg"
fi

echo "== assets"
ASSETS="$(dirname "$PROJECT")/assets"
if [ -d "$ASSETS" ]; then
  echo "   $(ls "$ASSETS" | wc -l) asset files in $ASSETS"
  cd "$PROJECT"
  "$WORK/venv/bin/python" -m synth.prep.asset_ingest
  "$BLENDER" -b --factory-startup -P synth/blender/probe_assets.py
  "$BLENDER" -b --factory-startup -P synth/blender/build_template.py
else
  echo "   no local asset directory (../assets): planning and tests work; building scenes needs the assets"
fi

echo "== tests"
cd "$PROJECT" && "$WORK/venv/bin/python" -m pytest tests -q || true
echo
echo "Done. In each new shell:  export BLENDER=$BLENDER"
