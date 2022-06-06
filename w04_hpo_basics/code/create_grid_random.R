#!/usr/bin/env Rscript

library(ggplot2)

plotTune = function(d) {
    d$TestAccuracy = mvtnorm::dmvnorm(x = d, mean = c(5,5), sigma = 40 * diag(2)) * 120 + 0.4
    pl = ggplot(data = d, aes(x = x, y = y, color = TestAccuracy))
    pl = pl + geom_point(size = d$TestAccuracy * 4)
    pl = pl + xlab("Hyperparameter 1") + ylab("Hyperparameter 2") + coord_fixed() + theme_minimal()
    return(pl)
}

x = y = seq(-10, 10, length.out = 10)
d = expand.grid(x = x, y = y)
pl = plotTune(d)
ggsave("../images/grid.png", pl)


x = runif(40, -10, 10)
y = runif(40, -10, 10)
d = data.frame(x = x, y = y)
pl = plotTune(d)
ggsave("../images/random.png", pl)
