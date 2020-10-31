#!/bin/sh
# could also be solved using a unified template for individual tex files but well...
pdftk t00_main.pdf cat 3-9 output w07_grey_t01_intro.pdf
pdftk t00_main.pdf cat 11-25 output w07_grey_t02_meta_learning.pdf
pdftk t00_main.pdf cat 27-35 output w07_grey_t03_mf_overview.pdf
pdftk t00_main.pdf cat 37-42 output w07_grey_t04_hyperband.pdf
pdftk t00_main.pdf cat 44-58 output w07_grey_t05_bohb.pdf
pdftk t00_main.pdf cat 60-66 output w07_grey_t06_mfbo.pdf
pdftk t00_main.pdf cat 68-81 output w07_grey_t07_lc_prediction.pdf
pdftk t00_main.pdf cat 83-93 output w07_grey_t08_success_stories.pdf
