# Vizualization of NSGAII

library(smoof)
library(ggplot2)
library(ecr)
library(gridExtra)
library(grid)

# viz in objective space 2 crit
plotObjectiveSpace2Crit = function(smoof.fun) {
  des = generateRandomDesign(n = 10000L, par.set = getParamSet(smoof.fun))
  des.eval = apply(des, 1, smoof.fun)
  des.eval.df = data.frame(t(des.eval))
  names(des.eval.df) = c("c1", "c2")
  
  p = ggplot() + geom_point(data = des.eval.df, aes(x = c1, y = c2), size = 0.7, color = "grey", alpha = 0.2)
  
  p = p + theme_bw()
  return(p)
}

# Example function
ps = makeNumericParamSet("x", lower = c(0.1, 0), upper = c(1, 5))

fn = makeMultiObjectiveFunction(name = "Min-Ex",
  fn = function(x) c(x[1], (1 + x[2]) / (x[1] * 60)),
  par.set = ps 
)

lower = getLower(ps)
upper = getUpper(ps)

MU = 25L
LAMBDA = 20L
mutator = setup(mutPolynomial, eta = 25, p = 0.2, lower = lower, upper = upper)
recombinator = setup(recSBX, eta = 15, p = 0.7, lower = lower, upper = upper)

set.seed(1)
res = ecr(fitness.fun = fn, lower = lower, upper = upper, mu = MU, lambda = LAMBDA, representation = "float", survival.strategy = "plus", 
  parent.selector = selSimple, mutator = mutator, 
  recombinator = recombinator, survival.selector = selNondom, 
  log.pop = TRUE, terminators = list(stopOnIters(max.iter = 10L)))

p = plotObjectiveSpace2Crit(fn)
populations = getPopulations(res$log)

for (i in c(1, 3, 5, 10)) {
  popdf = data.frame(t(populations[[i]]$fitness))
  pl = p + geom_point(data = popdf, aes(x = X1, y = X2), colour = "blue")
  pl = pl + ggtitle(paste("Iteration", i))
  assign(paste("p", i, sep = ""), value = pl)
}

g = grid.arrange(p1, p3, p10, ncol = 3)

ggsave(grid.draw(g), file = "images/NSGA2_steps.png", width = 8, height = 4)


# non-dominated sorting

pop = populations[[1]]$fitness
sorted = doNondominatedSorting(pop)
rank_max = max(sorted$ranks)
ranks = 1:rank_max

popdf = data.frame(t(pop))
popdf$Front = factor(sorted$ranks, ordered = TRUE, levels = ranks)


pl = p + geom_point(data = popdf[popdf$Front %in% ranks, ], aes(x = X1, y = X2, colour = Front)) 
pl = pl + geom_line(data = popdf[popdf$Front %in% ranks, ], aes(x = X1, y = X2, colour = Front), lty = 2)
ggsave(pl, file = "images/NSGA2_NDS.png", width = 4, height = 3)


# Crowd Sort - Example 1
F3 = popdf[which(popdf$Front == rank_max), ]
cd = computeCrowdingDistance(t(as.matrix(F3[, c("X1", "X2")])))

pl = p + geom_point(data = F3, aes(x = X1, y = X2), alpha = 0.3)
pl = pl + geom_line(data = F3, aes(x = X1, y = X2), lty = 2, alpha = 0.3)

  
pl1 = pl + geom_point(data = F3[order(cd, decreasing = FALSE)[1:5], ], aes(x = X1, y = X2), size = 3, shape = 17) 
pl1 = pl1 + theme(legend.position = "none")
pl2 = pl + geom_point(data = F3[order(cd, decreasing = TRUE)[1:5], ], aes(x = X1, y = X2), shape = 17, size = 3) 
pl2 = pl2 + theme(legend.position = "none")
pl2 = pl2 

g = grid.arrange(pl1, pl2, ncol = 2)

ggsave(grid.draw(g), file = "images/NSGA2_CS1.png", width = 6, height = 3)


cdo = order(cd, decreasing = TRUE)[c(5, length(cd)-1)]
F3.oX1 = F3[order(F3$X1), c("X1", "X2")]

cuboids = F3[cdo, c("X1", "X2")]
idx = which(F3.oX1$X1 %in% cuboids$X1)
cuboids = cbind(cuboids, F3.oX1[idx + 1, ])
cuboids = cbind(cuboids, F3.oX1[idx - 1, ])
names(cuboids) = c("x", "y", "xmin", "ymin", "xmax", "ymax")
cuboids$point = c("i", "j")

F3 = F3[!is.na(F3$X1), ]
pl1 = p + geom_point(data = F3, aes(x = X1, y = X2)) 
pl1 = pl1 + theme(legend.position = "none")
pl1 = pl1 + geom_line(data = F3, aes(x = X1, y = X2), lty = 2)
pl1 = pl1 + geom_rect(data = cuboids, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, colour = point, fill = point), alpha = 0.2)
pl1 = pl1 + geom_point(data = cuboids, aes(x = x, y = y, colour = point, fill = point), size = 3)

ggsave(pl1, file = "images/NSGA2_CS2.png", width = 3, height = 3)




