#!/bin/sh
# could also be solved using a unified template for individual tex files but well...
pdftk t00_main.pdf cat 4-18 output w06_bo_t01_overview.pdf
pdftk t00_main.pdf cat 20-46 output w06_bo_t02_cheaacq.pdf
pdftk t00_main.pdf cat 48-71 output w06_bo_t03_expacq.pdf
pdftk t00_main.pdf cat 73-91 output w06_bo_t04_surrogate.pdf
pdftk t00_main.pdf cat 93-105 output w06_bo_t05_extensions.pdf
pdftk t00_main.pdf cat 107-118 output w06_bo_t06_tpe.pdf
pdftk t00_main.pdf cat 120-131 output w06_bo_t07_stories.pdf
