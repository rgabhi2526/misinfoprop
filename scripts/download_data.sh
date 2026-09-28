#!/usr/bin/env bash
# Download the datasets into the paths the loaders expect (data_final/DATASETS.md).
# Idempotent: skips files that already exist (non-empty), resumes partial downloads.
#
#   scripts/download_data.sh            # final datasets: timme + pokec (~0.6GB download)
#   scripts/download_data.sh pokec      # one dataset: timme|pokec|reddit|gab
#   scripts/download_data.sh reserve    # reserve datasets: reddit + gab (gab 6GB, ~48GB unpacked)
#   CONN=32 scripts/download_data.sh    # more parallel streams per file (default 16)
#
# Sources:
#   timme   github.com/PatriciaXiao/TIMME (data in-repo)  -> data_final/timme/repo/data/
#   pokec   SNAP  snap.stanford.edu/data/soc-Pokec.html    -> data/soc-pokec-*.txt
#   reddit  SNAP  snap.stanford.edu/data/soc-RedditHyperlinks.html (reserve)
#   gab     Zenodo 10.5281/zenodo.1418347 (reserve)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$REPO/data_final"
CONN="${CONN:-16}"

dl() { # dl <url> <dest> — resume-safe, multi-stream when aria2c is installed
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "  ✓ $(basename "$dest") (exists)"; return; fi
  echo "  ↓ $(basename "$dest")"
  mkdir -p "$(dirname "$dest")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x"$CONN" -s"$CONN" -k1M --max-tries=5 --retry-wait=5 \
           --console-log-level=warn --summary-interval=15 \
           -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
  else
    curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  fi
}

need() { # need <file>... — fail loudly if a loader input is missing
  for f in "$@"; do [ -s "$f" ] || { echo "  ✗ missing $f"; exit 1; }; done
}

get_timme() {
  echo "== timme =="
  local r="$ROOT/timme/repo"
  if [ -d "$r/.git" ]; then
    echo "  ✓ repo (exists)"
  else
    # ponytail: only the three folders train_timme.py reads, not the whole repo
    git clone --depth 1 --filter=blob:none --sparse https://github.com/PatriciaXiao/TIMME.git "$r"
    git -C "$r" sparse-checkout set data/P_all data/formatted_location data/additional_labels
  fi
  need "$r/data/P_all/all_twitter_ids.csv" "$r/data/P_all/dict.csv" \
       "$r/data/P_all/friend_list.csv" "$r/data/formatted_location/simplified_user_info.json" \
       "$r/data/additional_labels/new_dict_cleaned.csv"
}

get_pokec() {
  echo "== pokec =="
  local d="$REPO/data"
  for f in soc-pokec-profiles soc-pokec-relationships; do
    if [ -s "$d/$f.txt" ]; then echo "  ✓ $f.txt (exists)"; continue; fi
    dl "https://snap.stanford.edu/data/$f.txt.gz" "$d/$f.txt.gz"
    echo "  ⇲ unpacking $f.txt"
    gunzip "$d/$f.txt.gz"
  done
  need "$d/soc-pokec-profiles.txt" "$d/soc-pokec-relationships.txt"
}

get_reddit() {
  echo "== reddit (reserve) =="
  local base="https://snap.stanford.edu/data" d="$ROOT/reddit-hyperlinks"
  dl "$base/soc-redditHyperlinks-body.tsv"       "$d/soc-redditHyperlinks-body.tsv"
  dl "$base/soc-redditHyperlinks-title.tsv"      "$d/soc-redditHyperlinks-title.tsv"
  dl "$base/web-redditEmbeddings-subreddits.csv" "$d/web-redditEmbeddings-subreddits.csv"
}

get_gab() {
  echo "== gab (reserve) =="
  local d="$ROOT/gab" tgz="$ROOT/gab/gab_posts_jan_2018.json.tar.gz"
  dl "https://zenodo.org/records/1418347/files/gab_posts_jan_2018.json.tar.gz?download=1" "$tgz"
  if [ ! -s "$d/extracted/gab_posts_jan_2018.json" ]; then
    echo "  ⇲ extracting (~48GB unpacked)…"
    mkdir -p "$d/extracted"
    tar -xzf "$tgz" -C "$d/extracted"
  else
    echo "  ✓ extracted (exists)"
  fi
}

case "${1:-final}" in
  timme)   get_timme ;;
  pokec)   get_pokec ;;
  reddit)  get_reddit ;;
  gab)     get_gab ;;
  final)   get_timme; get_pokec ;;
  reserve) get_reddit; get_gab ;;
  *) echo "usage: $0 [final|reserve|timme|pokec|reddit|gab]"; exit 1 ;;
esac
echo "done"
