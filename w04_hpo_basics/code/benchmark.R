# if (!exists("TunerCMAES", where = "package:mlr3tuning")) {
#   remotes::install_github("mlr-org/bbotk")
#   remotes::install_github("mlr-org/mlr3tuning@bbotk_cmaes")
# }

library(mlr3)
library(mlr3misc)
library(mlr3learners)
library(mlr3tuning)
library(cmaes)
library(xgboost)
library(paradox)
library(future)
library(future.batchtools)
library(batchtools)
library(ggplot2)
library(stringi)
library(gridExtra)
library(data.table)
library(checkmate)

set.seed(1)

#define 1! leaners
learner = lrn("classif.svm", predict_type = "prob", cost = 1, gamma = 1, type = "C-classification", kernel = "radial")
#define 1! terminator
n_evals = 100
terminator = term("evals", n_evals = n_evals)
#define n tuners
tuners = tnrs(c("grid_search", "random_search", "cmaes"))
tuners[[1]]$param_set$values$resolution = ceiling(sqrt(n_evals))
#define m tasks
tasks = tsks(c("spam", "sonar"))

ps = ParamSet$new(params = list(
    ParamDbl$new("cost", lower = -3, upper = 3),
    ParamDbl$new("gamma", lower = -3, upper = 3)
))

ps$trafo = function(x, param_set) {
  lapply(x, function(x) 10^x)
}

rsmp_tuning =  rsmp("cv", folds = 5)
#rsmp_tuning = rsmp("cv", folds = 2)

rsmp_outer = rsmp("cv", folds = 10)
#rsmp_outer = rsmp("cv", folds = 2)

learners = Map(function(ps, tuner) {
  AutoTuner$new(learner = learner, resampling = rsmp_tuning, measures = msr("classif.auc"), terminator = terminator, search_space = ps, tuner = tuner)  
}, ps = list(ps), tuners)

#add baseline
learner_default = lrn(learner$id, predict_type = learner$predict_type)
learner_default$id = paste0(learner_default$id, ".default")

#build design
design = benchmark_grid(tasks = tasks, learners = c(learners, learner, learner_default), resamplings = rsmp_outer)


#init inner resampling for all AutoTuners #FIXME Does not work because we would have to do it for the internal split
# design[, task_hash := map_chr(task, function(x) x$hash)]
# init_autotuner = function(task, at_learner) {
#   resampling = at_learner[[1]]$instance_args$resampling #already cloned
#   resampling$instantiate(task[[1]]) #all tasks are the same
#   for (lrn in at_learner) {
#     lrn$instance_args$resampling = resampling$clone()
#   }
#   return(list(at_learner))
# }
# design[map_lgl(learner, inherits, "AutoTuner"), learner := init_autotuner(.SD$task, .SD$learner), by = task_hash]
# design$task_hash = NULL

#init parallelization
reg_dir = if (fs::file_exists("~/nobackup/")) "~/nobackup/w04_hpo_benchmark" else "w04_hpo_basics/code/benchmark_bt"
unlink(reg_dir, recursive = TRUE)
reg = makeRegistry(file.dir = reg_dir, seed = 1)

batchMap(function(...) {
  set.seed(1) #makes inner resampling folds the same?
  #future::plan(multiprocess)
  res = benchmark(...)
  for (i in seq_row(res$data)) {
    if(!is.null(res$data$learner[[i]]$tuning_instance)) {
      res$data$learner[[i]]$tuning_instance$archive$data[,resample_result := NULL]
    }
  }
  return(res)
}, store_models = TRUE, design = split(design, seq_row(design)))

submitJobs(resources = list(ncpus = rsmp_outer$param_set$values$folds %??% 10))
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
