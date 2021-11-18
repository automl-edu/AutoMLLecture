#!/bin/sh

# Recommended entrypoint script:
# https://docs.github.com/en/actions/creating-actions/dockerfile-support-for-github-actions#example-entrypointsh-file

sh -c "pdftk $1 cat output $2"