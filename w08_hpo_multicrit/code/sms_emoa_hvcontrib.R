library(ggplot2)


# ref_p = c(9,9)
dd = data.frame(c1 = c(0, 2, 3, 8, 9), c2 = c(7, 5.5, 4 , 2, 9))
dd$c1_max = dd$c1
dd$c2_max = dd$c2
n = nrow(dd)

oo = order(dd$c1, decreasing = TRUE)
print(oo)
for (j in 2:n)
  dd[oo[j], "c1_max"] = dd[oo[j-1], "c1"]
oo = order(dd$c2, decreasing = TRUE)
print(oo)
for (j in 2:n)
  dd[oo[j], "c2_max"] = dd[oo[j-1], "c2"]


dd$hvc = (dd$c2_max - dd$c2) * (dd$c1_max - dd$c1)
dd$hvc[n] = Inf
j_worst = which.min(dd$hvc)

pl = ggplot(dd, aes(x = c1, y = c2))
pl = pl + geom_point(size = 3)
pl = pl + geom_rect(aes(xmin = c1, xmax = c1_max, ymin = c2, ymax = c2_max), alpha = 0.2)
# pl = pl + geom_point(x = 9, y = 9, size = 3)
# pl = pl + xlim(0,9) + ylim(0, 9)
# pl = pl + geom_text(aes(x = 5, y = 7, label = expression(lambda)))
pl = pl + geom_text(data = dd[-n,], aes(x = c1+0.7, y = c2+0.2, label = paste("HVC =",hvc)))
pl = pl +  annotate("text", x = dd$c1[j_worst]-0.2, y = dd$c2[j_worst]-0.2, size = 7, parse = TRUE, label = as.character(expression(tilde(lambda))))
print(pl)


