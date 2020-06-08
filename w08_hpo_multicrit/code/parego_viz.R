library(ggplot2)
library(ggpubr)


x = seq(0,3,length.out = 100)

fun2 = function(x) 3 * (x - 2)^2
fun1 = function(x) (x - 1)^2

tscheb = function(y1, y2, w1, w2, rho = 0.05) {
  max(y1*w1, y2*w2) + rho*(y1*w1 + y2*w2)
}

plotSpace = function(w1, w2) {


y1 = fun1(x)
y2 = fun2(x)

x = expand.grid(y1=y1, y2 = y2)


 x$c = mapply(tscheb, y1 = x$y1, y2 = x$y2, MoreArgs = list(w1 = w1, w2 = w2))
  ggplot(x, aes(x = y1, y = y2, fill = c, z = c)) + geom_tile() + geom_contour() + theme(legend.position = "None") + scale_fill_gradientn(colours=c("yellow","red")) + theme_bw() + xlab("c1") + ylab("c2") + ggtitle(paste("w1 =", w1, ", w2 =", w2))
}

p = ggarrange(plotSpace(0.9,0.1), plotSpace(0.4,0.6), plotSpace(0.1,0.9), common.legend = TRUE, legend = "bottom", nrow = 1)
