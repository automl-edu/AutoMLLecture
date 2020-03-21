library(mlr3)
library(mlr3misc)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(future)
library(future.batchtools)

#define 1! leaners
learner = lrn("classif.ranger", predict_type = "prob")
#define n tuners
tuners = tnrs(c("grid_search", "random_search", "gensa"))
#define 1! terminator
terminator = term("evals", n_evals = 50)
#define m tasks
tasks = tsks(c("spam", "sonar"))
#define m paramsets for each task in same order
paramsets = list(
  spam = ParamSet$new(params = list(
    ParamInt$new("mtry", 1, 58),
    ParamInt$new("min.node.size", 2, 20)
  )),
  sonar = ParamSet$new(params = list(
    ParamInt$new("mtry", 1, 61),
    ParamInt$new("min.node.size", 2, 20)
  ))
)
#matching paramsets and tasks
ptdt = data.table(paramset = paramsets, task = tasks)
#building all combinations
des = merge.data.frame(ptdt, data.table(tuner = tuners))
setDT(des)
#creating autotuner learner
des$learner = Map(function(ps, tuner) {
  AutoTuner$new(learner = learner, resampling = rsmp("cv", folds = 5), measures = msr("classif.auc"), terminator = terminator, tune_ps = ps, tuner = tuner)  
}, des$paramset, des$tuner)
#add outer cv foll all
des$resampling = Map(function(task) {
  resampling = rsmp("holdout", ratio = 1)
  resampling$instantiate(task = task)
  return(resampling)
  }, task = des$task)
#init parallelization
plan(multicore)
#run benchmark
bres = benchmark(des[,.(task, learner, resampling)], store_models = TRUE)
