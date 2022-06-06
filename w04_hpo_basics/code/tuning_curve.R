#!/usr/bin/env Rscript

library(ggplot2)

tr = BBmisc::load2("tune_example.RData")
df = as.data.frame(tr$opt.path)
ggd = df[, c("dob", "auc.test.mean")]
colnames(ggd) = c("iter", "auc")
ggd$auc = cummax(ggd$auc)
pl = ggplot(ggd, aes(x = iter, y = auc))
pl = pl + geom_line()
pl = pl + theme_bw()
pl = pl +
  theme(axis.text=element_text(size=18), axis.title=element_text(size=22)) +
  ylab("Maximal AUC") + xlab("Iterations") + theme_minimal()

ggsave("../images/curve.png", pl)
