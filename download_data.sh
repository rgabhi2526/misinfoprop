#!/usr/bin/env bash
# Download every dataset into data_final/. Idempotent: skips files that already
# exist (non-empty) and resumes partial downloads (curl -C -). Re-run any time.
#
#   ./download_data.sh            # everything
#   ./download_data.sh reddit     # one dataset: timme|reddit|voterfraud|gab
#   CONN=32 ./download_data.sh    # more parallel streams per file (default 16)
#   # different servers at once:  ./download_data.sh gab & ./download_data.sh voterfraud &
#
# Sources:
#   timme       github.com/PatriciaXiao/TIMME (data is in-repo)
#   reddit      SNAP  snap.stanford.edu/data/soc-RedditHyperlinks.html
#   voterfraud  Figshare 10.6084/m9.figshare.13571084
#   gab         Zenodo  10.5281/zenodo.1418347  (6GB, md5 below)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)/data_final"
mkdir -p "$ROOT"

# Faster downloads: aria2c opens CONN parallel streams per file (segmented,
# resume-safe). Install it (`brew install aria2` / `apt install aria2`) to use
# it; otherwise we fall back to single-stream curl. Override streams with
# CONN=32 ./download_data.sh
CONN="${CONN:-16}"

dl() { # dl <url> <dest>  — resume-safe, multi-connection when aria2c is present
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "  ✓ $(basename "$dest") (exists)"; return; fi
  echo "  ↓ $(basename "$dest")"
  mkdir -p "$(dirname "$dest")"
  # Figshare blocks aria2 (403 on its User-Agent + flaky async 202), so those
  # files always go through curl. aria2 (multi-stream) is used elsewhere.
  if command -v aria2c >/dev/null 2>&1 && [[ "$url" != *figshare* ]]; then
    aria2c -c -x"$CONN" -s"$CONN" -k1M --max-tries=5 --retry-wait=5 \
           --console-log-level=warn --summary-interval=15 \
           -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
  else
    curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
  fi
}

get_timme() {
  echo "== timme =="
  if [ -d "$ROOT/timme/repo/.git" ]; then
    echo "  ✓ repo (exists)"
  else
    git clone --depth 1 https://github.com/PatriciaXiao/TIMME.git "$ROOT/timme/repo"
  fi
}

get_reddit() {
  echo "== reddit =="
  local base="https://snap.stanford.edu/data"
  dl "$base/soc-redditHyperlinks-body.tsv"        "$ROOT/reddit-hyperlinks/soc-redditHyperlinks-body.tsv"
  dl "$base/soc-redditHyperlinks-title.tsv"       "$ROOT/reddit-hyperlinks/soc-redditHyperlinks-title.tsv"
  dl "$base/web-redditEmbeddings-subreddits.csv"  "$ROOT/reddit-hyperlinks/web-redditEmbeddings-subreddits.csv"
}

get_voterfraud() {
  echo "== voterfraud =="
  local d="$ROOT/voterfraud2020" f="https://ndownloader.figshare.com/files"
  dl "$f/26069594" "$d/tweets-2020-10.csv"
  dl "$f/26069630" "$d/tweets-2020-11.csv"
  dl "$f/26069609" "$d/tweets-2020-12.csv"
  dl "$f/26069735" "$d/retweets-2020-10.csv"
  dl "$f/26069741" "$d/retweets-2020-11.csv"
  dl "$f/26069747" "$d/retweets-2020-12.csv"
  dl "$f/26069726" "$d/users.csv"
  dl "$f/26069783" "$d/urls.csv"
  dl "$f/26069807" "$d/images.csv"
  dl "$f/26069780" "$d/youtube_videos.csv"
}

get_gab() {
  echo "== gab =="
  local d="$ROOT/gab" tgz="$ROOT/gab/gab_posts_jan_2018.json.tar.gz"
  dl "https://zenodo.org/records/1418347/files/gab_posts_jan_2018.json.tar.gz?download=1" "$tgz"
  # ponytail: md5 check optional; uncomment to verify the 6GB download
  # echo "e6322adb28ab7cd499826c940a66685f  $tgz" | md5sum -c -
  if [ ! -s "$d/extracted/gab_posts_jan_2018.json" ]; then
    echo "  ⇲ extracting (large, ~11GB unpacked)…"
    mkdir -p "$d/extracted"
    tar -xzf "$tgz" -C "$d/extracted"
  else
    echo "  ✓ extracted (exists)"
  fi
}

case "${1:-all}" in
  timme)      get_timme ;;
  reddit)     get_reddit ;;
  voterfraud) get_voterfraud ;;
  gab)        get_gab ;;
  all)        get_timme; get_reddit; get_voterfraud; get_gab ;;
  *) echo "usage: $0 [timme|reddit|voterfraud|gab|all]"; exit 1 ;;
esac
echo "done → $ROOT"
