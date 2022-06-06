#!/usr/bin/env Rscript

library(ggplot2)
theme_set(theme_minimal())
library(smoof)
library(ecr)

MU = 20L; LAMBDA = 5L; MAX.ITER = 200L
lower = - 30
upper = 30

#notation
lab_xx = expression(bold(lambda))
lab_y = expression(c(bold(lambda)))

#save images
mysave = function(name, ...) {
  ggsave(paste0("../images/",name,".png"), height = 7, width = 14)
}

fn = makeAckleyFunction(1L)

control = initECRControl(fn)
control = registerECROperator(control, "mutate", mutGauss, sdev = 2, lower = lower, upper = upper)
control = registerECROperator(control, "selectForSurvival", selGreedy)

pl = autoplot(fn, show.optimum = F, length.out = 1000L)
pl = pl + labs(x = lab_xx, y = lab_y)

mysave("ea_ex1")

set.seed(1234)

population = genReal(MU, getNumberOfParameters(fn), lower, upper)
fitness = evaluateFitness(control, population)

pl = autoplot(fn, show.optimum = F, length.out = 1000L)
pl = pl + labs(x = lab_xx, y = lab_y)
df = data.frame(x = unlist(population), y = as.numeric(fitness))
pl = pl + geom_point(data = df, mapping = aes(x = x, y = y), size = 3)

mysave("ea_ex2")

# neutrale Selektion von lambda Eltern
set.seed(1234)
idx = sample(1:MU, LAMBDA)


pl = pl + geom_point(data = df[idx, ], mapping = aes(x = x, y = y), colour = "red", size = 3)
mysave("ea_ex3")

offspring = mutate(control, population[idx], p.mut = 1)
fitness.o = evaluateFitness(control, offspring)
df.o = data.frame(x = unlist(offspring), y = as.numeric(fitness.o))

pl = pl + geom_point(data = df.o, aes(x = x, y = y), color = "red", size = 3)
pl = pl + geom_point(data = df[idx,], aes(x = x, y = y), color = "red", size = 3)
pl2 = pl + geom_segment(data = data.frame(x = df[idx, ]$x, y = df[idx, ]$y, xend = df.o$x, yend = df.o$y), aes(x = x, y = y, xend = xend, yend = yend), colour = "red", linetype = 1, arrow = arrow(length = unit(0.01, "npc"))
)

mysave("ea_ex4", pl2)

sel = replaceMuPlusLambda(control, population, offspring, fitness, fitness.o)
population = sel$population
fitness = sel$fitness
df = data.frame(x = unlist(population), y = as.numeric(fitness))

pl = pl + geom_point(data = df, aes(x = x, y = y), color = "green", fill = "green", size = 3)
pl = pl + geom_hline(yintercept = max(df$y), lty = 2)

mysave("ea_ex5")
