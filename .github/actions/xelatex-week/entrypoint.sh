#!/bin/sh

# Recommended entrypoint script:
# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#example-entrypointsh-file

cd $1
sh -c "mkdir $2";
for f in *.tex ; do
  sh -c "xelatex -interaction=nonstopmode -jobname=${f%.*} \"\def\is$2{} \input{$f}\"" ;
  sh -c "mv ${f%.*}.pdf ${2}/${f%.*}.pdf";
done
