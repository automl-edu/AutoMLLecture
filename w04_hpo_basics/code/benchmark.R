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
if (!fs::file_exists("benchmark_res.rds")) {
  reg_dir = if (fs::file_exists("~/nobackup/")) "~/nobackup/w04_hpo_benchmark" else "w04_hpo_basics/code/benchmark_bt"
  unlink(reg_dir, recursive = TRUE)
  reg = makeRegistry(file.dir = reg_dir, seed = 1, packages = c("mlr3", "mlr3tuning", "mlr3misc", "stringi", "future"))
  
  batchMap(function(design, ...) {
    #makes inner resampling folds the same if the outer resampling is the same?
    set.seed(as.integer(substr(stri_replace_all_regex(design$resampling[[1]]$hash, "[a-z]", ""),0,9)))
    future::plan(multiprocess)
    res = benchmark(design = design, ...)
    for (i in seq_row(res$data)) {
      if(!is.null(res$data$learner[[i]]$tuning_instance)) {
        res$data$learner[[i]]$tuning_instance$archive$data[,resample_result := NULL]
      }
    }
    return(res)
  }, store_models = TRUE, design = split(design, seq_row(design)))
  
  #testJob(1)
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
  
  
  saveRDS(list(baseline = res_baseline, tune = res_tune), "benchmark_res.rds")
}

res = readRDS("benchmark_res.rds")

baseline_res = res$baseline$aggregate(measures = msr("classif.auc"))
baseline_res$tuner = "untuned"

#build dt for plotting
res_compl = res$tune$data[, list(opt_path = {
  x = learner[[1]]
  list(cbind(x$archive$data, tuner = class(x$tuner)[1], task_id = x$model$tuning_instance$objective$task$id, learner_id = x$learner$id, nr = seq_row(x$archive$data)))
  }), by = .(uhash, iteration)]
res_compl = setDT(tidyr::unnest(res_compl, "opt_path")) #tidyr::unnest can deal with data.frames to be unnested
res_compl = unnest(res_compl, "opt_x", prefix = "opt.x.")
 
res_compl[, classif.auc.cummax := cummax(classif.auc), by = .(task_id, learner_id, tuner, uhash, iteration)]
res_compl[, tuner := stri_replace_first_fixed(tuner, "Tuner", "")]
res_compl = res_compl[nr <= 100,]

theme_set(theme_bw())

tuner_names = c("GridSearch", "RandomSearch", "CMAES", "Untuned", "Heuristic")
tuner_colors = set_names(RColorBrewer::brewer.pal(7, "Set1"), tuner_names)

res_compl[, tuner := factor(tuner, levels = tuner_names)]

#tune curve for iter = 1
g = ggplot(res_compl[iteration == 1,], aes(y = classif.auc.cummax, x = nr, color = tuner))
g = g + geom_line()
g = g + geom_point(aes(y = classif.auc), alpha = 0.1)
g = g + facet_wrap("task_id")
g = g + coord_cartesian(ylim = c(0.5, 1))
g = g + scale_color_manual(values = tuner_colors)
g = g + labs(y = "AUC", x = "eval", title = "Tuning cost and gamma for SVM (kernel = radial)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_curve_iter_1.png", g, height = 5, width = 10)

#tune curve for all iters
g = ggplot(res_compl, aes(y = classif.auc.cummax, x = nr, color = tuner, group = paste0(tuner, iteration)))
g = g + geom_line()
g = g + facet_wrap("task_id")
g = g + coord_cartesian(ylim = c(0.5, 1))
g = g + scale_color_manual(values = tuner_colors)
g = g + labs(y = "AUC", x = "eval", title = "Tuning cost and gamma for SVM (kernel = radial)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_curve_iter_all.png", g, height = 5, width = 10)

#tune curve for all iters averaged
g = ggplot(res_compl, aes(y = classif.auc.cummax, x = nr, color = tuner))
g = g + stat_summary(geom = "line", fun = median)
g = g + facet_wrap("task_id")
g = g + coord_cartesian(ylim = c(0.9, 1))
g = g + scale_color_manual(values = tuner_colors)
g = g + labs(y = "AUC", x = "eval", title = "Tuning cost and gamma for SVM (kernel = radial)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_curve_median.png", g, height = 5, width = 10)

#tune curve for all iters averaged + individual
g = ggplot(res_compl, aes(y = classif.auc.cummax, x = nr, color = tuner))
g = g + geom_line(alpha = 0.2, mapping = aes(group = paste0(tuner, iteration)))
g = g + stat_summary(geom = "line", fun = median)
g = g + facet_wrap("task_id")
g = g + coord_cartesian(ylim = c(0.9, 1))
g = g + scale_color_manual(values = tuner_colors)
g = g + labs(y = "AUC", x = "eval", title = "Tuning cost and gamma for SVM (kernel = radial)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_curve_iter_all_median.png", g, height = 5, width = 10)

# outer performance
res_outer = res$tune$score(measures = msr("classif.auc"))
res_outer[, tuner := map_chr(learner, function(x) class(x$tuner)[[1]])]
res_outer[, tuner := stri_replace_first_fixed(tuner, "Tuner", "")]
res_baseline = res$baseline$score(measures = msr("classif.auc"))
res_baseline[, tuner := ifelse(stri_detect_fixed(learner_id, "default"), "Heuristic", "Untuned")]
res_combined = rbind(res_baseline, res_outer)
res_combined[, tuner:=factor(tuner, levels = tuner_names)]
settings = list(
  tuners = list(name = "tuners", tuners = unique(res_outer$tuner)),
  untuned = list(name = "default", tuners = c(unique(res_outer$tuner), "Untuned")),
  all = list(name = "all", tuners = unique(res_combined$tuner), ylim = c(0.8, 1))
)
for (s in settings) {
  g = ggplot(res_combined[tuner %in% s$tuners], aes(x = tuner, y = classif.auc, fill = tuner))
  g = g + geom_boxplot()
  g = g + scale_fill_manual(values = tuner_colors)
  g = g + facet_grid(task_id~.)
  g = g + coord_flip(ylim = s$ylim)
  if (interactive()) {
    print(g)
  }
  ggsave(sprintf("../images/benchmark_boxplot_%s.png", s$name), g, height = 5, width = 10)
}

#tune x space
gs = lapply(unique(res_compl$task_id), function(this_task_id) {
  g = ggplot(res_compl[task_id == this_task_id & iteration == 1], aes(x = opt.x.cost, y = opt.x.gamma, size = classif.auc, color = classif.auc))
  g = g + geom_point()
  g = g + facet_grid(task_id~tuner)
  g = g + scale_radius() + scale_colour_viridis_c() + scale_y_log10() + scale_x_log10()
  g = g + labs(color = "AUC", size = "AUC")
  g + theme_bw()
})
g = marrangeGrob(gs, ncol = 2, nrow = 1, top = "Tuning eta and lambda for xgboost (nrounds = 100)")
if (interactive()) {
  print(g)
}
ggsave("../images/benchmark_scatter.png", g, height = 5, width = 10)
