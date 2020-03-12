#!/usr/bin/env Rscript

library(ggplot2)
library(smoof)
library(ecr)

MU = 20L; LAMBDA = 5L; MAX.ITER = 200L
lower = - 30
upper = 30


fn = makeAckleyFunction(1L)

control = initECRControl(fn)
control = registerECROperator(control, "mutate", mutGauss, sdev = 2, lower = lower, upper = upper)
control = registerECROperator(control, "selectForSurvival", selGreedy)

pl = autoplot(fn, show.optimum = F, length.out = 1000L)
pl = pl + theme_minimal()

ggsave("../images/ea_ex1.png", pl, height = 7, width = 14)

set.seed(1234)

population = genReal(MU, getNumberOfParameters(fn), lower, upper)
fitness = evaluateFitness(control, population)

pl = autoplot(fn, show.optimum = F, length.out = 1000L)
df = data.frame(x = unlist(population), y = as.numeric(fitness))
pl = pl + geom_point(data = df, mapping = aes(x = x, y = y), size = 3) + theme_minimal()

ggsave("../images/ea_ex2.png", pl, height = 7, width = 14)

# neutrale Selektion von lambda Eltern
set.seed(1234)
idx = sample(1:MU, LAMBDA)


pl = pl + geom_point(data = df[idx, ], mapping = aes(x = x, y = y), colour = "red", size = 3)
ggsave("../images/ea_ex3.png", pl, height = 7, width = 14)

offspring = mutate(control, population[idx], p.mut = 1)
fitness.o = evaluateFitness(control, offspring)
df.o = data.frame(x = unlist(offspring), y = as.numeric(fitness.o))

pl = pl + geom_point(data = df.o, aes(x = x, y = y), color = "red", size = 3)
pl = pl + geom_point(data = df[idx,], aes(x = x, y = y), color = "red", size = 3)
pl2 = pl + geom_segment(data = data.frame(x = df[idx, ]$x, y = df[idx, ]$y, xend = df.o$x, yend = df.o$y), aes(x = x, y = y, xend = xend, yend = yend), colour = "red", linetype = 1, arrow = arrow(length = unit(0.01, "npc"))
)

ggsave("../images/ea_ex4.png", pl2, height = 7, width = 14)

sel = replaceMuPlusLambda(control, population, offspring, fitness, fitness.o)
population = sel$population
fitness = sel$fitness
df = data.frame(x = unlist(population), y = as.numeric(fitness))

pl = pl + geom_point(data = df, aes(x = x, y = y), color = "green", fill = "green", size = 3)
pl = pl + geom_hline(yintercept = max(df$y), lty = 2)

ggsave("../images/ea_ex5.png", pl, height = 7, width = 14)
