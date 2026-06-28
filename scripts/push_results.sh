#!/bin/bash
# Publish the eval CSVs to the git remote -- same pattern as the working figure job:
# copy the results into the repo, `git add` ONLY those files (never `git add -A`),
# commit, and `git push origin <current-branch>`. Relies on the node's existing git
# credentials (ssh key / credential helper), exactly like that job does. `unset
# DISPLAY` keeps a headless node from launching the gnome-ssh-askpass GUI.
#
#   push_results.sh <repo_dir> <results_dir> [in_repo_subdir]   (default subdir: results)
#
# The CSVs land in <repo>/<subdir>/ on the current branch; pull them with `git pull`.
set -uo pipefail

repo="${1:?repo dir}"; results="${2:?results dir}"; subdir="${3:-results}"
unset DISPLAY                  # no GUI askpass on a headless node
export GIT_TERMINAL_PROMPT=0   # fail fast instead of hanging on a credential prompt

cd "$repo"
ls "$results"/*.csv >/dev/null 2>&1 || { echo "[git] no CSVs in $results; skipping"; exit 0; }

mkdir -p "$subdir"
cp -f "$results"/*.csv "$subdir"/
git add "$subdir"/*.csv

if git diff --cached --quiet; then
  echo "[git] no CSV changes to commit"
else
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  git commit -q -m "eval results: job ${SLURM_JOB_ID:-local} ($(date -u +%Y-%m-%dT%H:%MZ))"
  git push origin "$BRANCH"
  echo "[git] pushed $(ls "$subdir"/*.csv | wc -l | tr -d ' ') CSVs to origin/$BRANCH"
fi
