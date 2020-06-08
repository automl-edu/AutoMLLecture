library(ggplot2)
library(ggpubr)
theme_set(theme_bw())

res = 100
x_range = c(0,3)

# we need an equidistant grid on y1,y2 for ggplot2, therefore we use inverted functions
# only works if fun1 and fun2 are not correlated
fun2 = function(x) 3 * (x - 2)^2
fun2_inv = function(y) 2 + sqrt(y)/sqrt(3) # only for x positive!

fun1 = function(x) (x - 1)^2
fun1_inv = function(y) 1 + sqrt(y) # only for x positive!

y1_range = fun1(x_range)
y2_range = fun2(x_range)

y1_seq = do.call(seq, c(as.list(y1_range), length.out = res))
y2_seq = do.call(seq, c(as.list(y2_range), length.out = res))

x1_seq = fun1_inv(y1_seq)
x2_seq = fun2_inv(y2_seq)

tscheb = function(y1, y2, w1, w2, rho = 0.05) {
  max(y1*w1, y2*w2) + rho*(y1*w1 + y2*w2)
}

xgrid = expand.grid(y1=y1_seq, y2=y2_seq)

plotSpace = function(w1, w2) {

  if (FALSE) {
    w1 = 0.1; w2 = 0.9
  }
  
  x = ygrid

  x$c = mapply(tscheb, y1 = x$y1, y2 = x$y2, MoreArgs = list(w1 = w1, w2 = w2))
  
  g = ggplot(x, aes(x = y1, y = y2, fill = c, z = c))
  g = g + geom_raster() + geom_contour()
  g = g + theme(legend.position = "None")
  g = g + scale_fill_gradientn(colours=c("yellow","red"))
  g = g + labs(x = "c1", y = "c2", title = paste("w1 =", w1, ", w2 =", w2))
  g
}

p = ggarrange(plotSpace(0.9,0.1), plotSpace(0.4,0.6), plotSpace(0.1,0.9), common.legend = TRUE, legend = "bottom", nrow = 1)
if (interactive()) print(p)
ggsave("../images/parego_viz.png", p, height = 5, width = 6)
