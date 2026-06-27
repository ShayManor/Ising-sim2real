#!/bin/bash
# Push eval result CSVs to a dedicated git branch so they can be pulled elsewhere.
#
#   push_results.sh <repo_dir> <results_dir> [branch]
#
# Results live outside the repo (e.g. on /scratch), so this stages them into a
# throwaway git worktree on a results-only branch, commits, and pushes -- without
# touching the job's main checkout. Accumulates across runs (commits on top of the
# remote branch). Always exits 0: a failed push must never fail the eval job; the
# CSVs remain in <results_dir> and can be pushed by hand.
#
# Pull them locally with:  git fetch origin && git switch <branch>   (default eval-results)
set -uo pipefail

repo="${1:?repo dir}"; results="${2:?results dir}"; branch="${3:-eval-results}"
export GIT_TERMINAL_PROMPT=0   # fail instead of hanging on a credential prompt

command -v git >/dev/null 2>&1 || { echo "[push] git not found; skipping"; exit 0; }
ls "$results"/*.csv >/dev/null 2>&1 || { echo "[push] no CSVs in $results; skipping"; exit 0; }

git -C "$repo" worktree prune 2>/dev/null || true
wt="$(mktemp -d)"
cleanup () { git -C "$repo" worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"; }
trap cleanup EXIT

# Base the worktree on the existing remote branch (accumulate) or current HEAD (first run).
git -C "$repo" fetch origin "$branch" 2>/dev/null || true
if git -C "$repo" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  git -C "$repo" worktree add -f "$wt" -B "$branch" "origin/$branch" || { echo "[push] worktree add failed; skipping"; exit 0; }
else
  git -C "$repo" worktree add -f "$wt" -b "$branch" || { echo "[push] worktree add failed; skipping"; exit 0; }
fi

mkdir -p "$wt/results"
cp -f "$results"/*.csv "$wt/results/"
git -C "$wt" add results

if git -C "$wt" diff --cached --quiet; then
  echo "[push] no CSV changes to push"
  exit 0
fi

git -C "$wt" \
  -c user.name="${GIT_AUTHOR_NAME:-ising-eval}" \
  -c user.email="${GIT_AUTHOR_EMAIL:-ising-eval@cluster}" \
  commit -q -m "eval results: job ${SLURM_JOB_ID:-local} $(date -u +%Y-%m-%dT%H:%MZ)"

if git -C "$wt" push -q origin "$branch"; then
  echo "[push] pushed $(ls "$results"/*.csv | wc -l | tr -d ' ') CSVs to origin/$branch"
else
  echo "[push] push FAILED (CSVs remain in $results; check git credentials on this node)"
fi
exit 0
