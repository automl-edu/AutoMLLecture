library(mlr3)
library(mlr3oml)
#library(mlr3viz)
task = tsk("oml", data_id = 1590)
#autoplot(task, type = "pairs", cardinality_threshold = 16)
#autoplot(task, type = "duo", cardinality_threshold = 16)
#mlr3viz:::autoplot.TaskClassif
#

# remotes::install_github("ggobi/ggally")
library(GGally)
library(ggplot2)
theme_set(theme_bw())
pdata = mlr3viz::fortify(task)

g = ggally_colbar(pdata, aes(x = race, y = class), size = 3, label_format = scales::label_percent(accuracy = 1))
g = g + facet_grid(sex~.)
g = g + theme(axis.text.x = element_text(angle = 45, hjust = 1))
g = g + theme(legend.position = "none")
if (interactive()) print(g)
ggsave("../images/dataset_adult_race.png", g, height = 6, width = 3)

g = ggally_colbar(pdata, aes(x = occupation, y = class), size = 3, label_format = scales::label_percent(accuracy = 1))
g = g + facet_grid(sex~.)
g = g + theme(axis.text.x = element_text(angle = 45, hjust = 1))
g = g + theme(legend.position = "none")
if (interactive()) print(g)
ggsave("../images/dataset_adult_education.png", g, height = 6, width = 5)

g = ggplot(pdata, aes(x=age, group=class, fill=class)) + geom_histogram(binwidth=1, color='black', size = 0.1)
if (interactive()) print(g)
ggsave("../images/dataset_adult_age_sex.png", g, height = 2.5, width = 5.5)

if (FALSE) {

  g = ggally_colbar(pdata, aes(x = sex, y = class))
  if (interactive()) print(g)
  ggsave("../images/dataset_adult_sex.png", g, height = 5, width = 7)
}

