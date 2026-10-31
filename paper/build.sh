#!/bin/sh
# Builds paper/main.pdf and stamps the cited code commit (\RepoCommit) from git.
# Marks the hash "-dirty" when the working tree has uncommitted changes.
cd "$(dirname "$0")" || exit 1
hash=$(git rev-parse --short HEAD)
[ -n "$(git status --porcelain --untracked-files=no)" ] && hash="$hash-dirty"
printf '%s\n' "\\renewcommand{\\RepoCommit}{$hash}" > commit.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex
