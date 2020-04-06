library(mlr3)
library(mlr3misc)
library(mlr3learners)
library(mlr3tuning)
library(GenSA)
library(xgboost)
library(paradox)
library(future)
library(future.batchtools)
library(batchtools)
library(ggplot2)
library(stringi)
library(gridExtra)

set.seed(1)

#define 1! leaners
learner = lrn("classif.xgboost", predict_type = "prob", nrounds = 100)
#define 1! terminator
n_evals = 100
terminator = term("evals", n_evals = n_evals)
#define n tuners
tuners = tnrs(c("grid_search", "random_search", "gensa"))
tuners[[1]]$param_set$values$resolution = ceiling(sqrt(n_evals))
#define m tasks
tasks = tsks(c("spam", "sonar"))

#define m paramsets for each task in same order

# Made sense for RF
# paramsets = list(
#   spam = ParamSet$new(params = list(
#     ParamInt$new("mtry", 1, 58),
#     ParamInt$new("min.node.size", 2, 20)
#   )),
#   sonar = ParamSet$new(params = list(
#     ParamInt$new("mtry", 1, 61),
#     ParamInt$new("min.node.size", 2, 20)
#   ))
# )

paramsets = list(
  ParamSet$new(params = list(
    ParamDbl$new("eta", lower = 0.1, upper = 0.5),
    ParamDbl$new("lambda", lower = - 1, upper = 0)
  ))
)
paramsets[[1]]$trafo = function(x, param_set) {
  x$lambda = 10^x$lambda
  x
}

#matching paramsets and tasks
ptdt = data.table(paramset = paramsets, task = tasks)
#building all combinations
des = merge.data.frame(ptdt, data.table(tuner = tuners))
setDT(des)
#creating autotuner learner
rsmp_tuning =  rsmp("cv", folds = 5)
des$learner = Map(function(ps, tuner) {
  AutoTuner$new(learner = learner, resampling = rsmp_tuning, measures = msr("classif.auc"), terminator = terminator, tune_ps = ps, tuner = tuner)  
}, des$paramset, des$tuner)
#init outer resampling for all
des$resampling = Map(function(task) {
  resampling = rsmp("holdout", ratio = 1)
  resampling$instantiate(task = task)
  return(resampling)
}, task = des$task)


design = des[,.(task, learner, resampling)]

#add baseline
baseline_design = benchmark_grid(tasks = tasks, learners = list(learner), resamplings = rsmp_tuning)
design = rbind(design, baseline_design)

#init parallelization
reg_dir = if (fs::dir_exists("~/nobackup/")) "~/nobackup/w04_hpo_benchmark" else "w04_hpo_basics/code/benchmark_bt"
reg = makeRegistry(file.dir = reg_dir)

batchMap(benchmark, store_models = TRUE, design = split(design, seq_row(design)))
submitJobs()
waitForJobs()
res = reduceResultsList()
res_tune = res[[1]]
res_baseline = NULL
for (i in 2:length(res)) {
  if (inherits(res[[i]]$learners$learner[[1]], "AutoTuner")) {
    res_tune$combine(res[[i]])
  } else if (is.null(res_baseline)) {
    res_baseline = res[[i]]
  } else {
    res_baseline$combine(res[[i]])
  }
}

baseline_res = res_baseline$aggregate(measures = msr("classif.auc"))
baseline_res$tuner = "untuned"

#build dt for plotting
res_compl = map_dtr(res_tune$data$learner, function(x) cbind(x$archive(), tuner = class(x$tuner)[1]))
res_compl[, classif.auc.cummax := cummax(classif.auc), by = .(task_id, learner_id, tuner)]
res_compl[, tuner := stri_replace_first_fixed(tuner, "Tuner", "")]
#unnest(res_compl, "params")

#tune curve
g = ggplot(res_compl, aes(y = classif.auc.cummax, x = nr, color = tuner))
g = g + geom_line()
g = g + geom_point(aes(y = classif.auc), alpha = 0.5)
g = g + geom_hline(data = baseline_res, mapping = aes(yintercept = classif.auc, color = tuner), lty = "dashed")
g = g + facet_wrap("task_id", scales = "free")
g = g + theme_bw()
g = g + labs(y = "AUC", x = "eval", title = "Tuning eta and lambda for xgboost (nrounds = 100)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_curve.png", g, height = 5, width = 10)

#tune x space
gs = lapply(unique(res_compl$task_id), function(this_task_id) {
  g = ggplot(res_compl[task_id == this_task_id], aes(x = eta, y = lambda, size = classif.auc, color = classif.auc))
  g = g + geom_point()
  g = g + facet_grid(task_id~tuner)
  g = g + scale_radius() + scale_colour_viridis_c() + scale_y_log10(breaks = c(0.1, 0.2, 0.4, 0.8, 1))
  g = g + labs(color = "AUC", size = "AUC")
  g + theme_bw()
})
g = marrangeGrob(gs, ncol = 2, nrow = 1, top = "Tuning eta and lambda for xgboost (nrounds = 100)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_scatter.png", g, height = 5, width = 10)
