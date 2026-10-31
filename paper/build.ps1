# Builds paper/main.pdf and stamps the cited code commit (\RepoCommit) from git.
# Marks the hash "-dirty" when the working tree has uncommitted changes.
Set-Location $PSScriptRoot
$hash = (git rev-parse --short HEAD).Trim()
if (git status --porcelain --untracked-files=no) { $hash += "-dirty" }
Set-Content -Path commit.tex -Value "\renewcommand{\RepoCommit}{$hash}" -Encoding ascii
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
