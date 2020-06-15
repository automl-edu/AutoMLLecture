library(mlr)
library(mlrMBO)
library(batchtools)
library(checkmate)

lrn = makeLearner("classif.svm", predict.type = "response") 
tsk = spam.task
n_evals = 100

make_mbo_multi_control = function(multi_method) {
  mbo_ctrl = makeMBOControl(n.objectives = 2)
  mbo_ctrl = setMBOControlTermination(mbo_ctrl, max.evals = n_evals)
  mbo_ctrl = setMBOControlMultiObj(mbo_ctrl, method = multi_method)
  if (multi_method == "dib") {
    mbo_ctrl = setMBOControlInfill(mbo_ctrl, crit = crit.dib1)
  }
  makeTuneMultiCritControlMBO(n.objectives = 2, mbo.control = mbo_ctrl)
}

tune_ctrls = list(
  grid = makeTuneMultiCritControlGrid(resolution = ceiling(sqrt(n_evals))),
  random = makeTuneMultiCritControlRandom(maxit = n_evals),
  nsga2 = makeTuneMultiCritControlNSGA2(budget = 12 * ceiling(n_evals / 12), popsize = 12, max.generations = ceiling(n_evals / 12)),
  mbo_parego = make_mbo_multi_control("parego"),
  mbo_dib = make_mbo_multi_control("dib")
)

ps = makeParamSet(
  makeNumericParam("cost", lower = -3, upper = 3, trafo = function(x) 10^x),
  makeNumericParam("gamma", lower = -3, upper = 3, trafo = function(x) 10^x)
)

rsmp_tuning = makeResampleDesc("CV", iters = 5)
rsmp_outer = makeResampleDesc("CV", iters = 10)

if (!fs::file_exists("benchmark_res.rds")) {
  reg_dir = if (fs::file_exists("~/nobackup/")) "~/nobackup/w08_multicrit_benchmark" else "benchmark_bt"
  unlink(reg_dir, recursive = TRUE)
  reg = makeExperimentRegistry(file.dir = reg_dir, seed = 1, packages = c("mlr", "mlrMBO", "parallelMap", "checkmate"))
  
  make_instance_gen = function(rsmp_outer) {
    force(rsmp_outer)
    instance_gen = function(data, job) {
      fold = job$repl 
      rsmpl_outer_fixed = makeResampleInstance(rsmp_outer, task = data$tsk)
      list(train = rsmpl_outer_fixed$train.inds[[fold]], test = rsmpl_outer_fixed$test.inds[[fold]], fold = fold)
    }
  }
  
  for (n in names(tune_ctrls)) {
    addProblem(name = n, data = list(lrn = lrn, tsk = tsk, rsmp_tuning = rsmp_tuning, ps = ps, ctrl = tune_ctrls[[n]]), fun = make_instance_gen(rsmp_outer), seed = 1)  
  }
  
  algo_mlr = function(job, data, instance) {
    lrn = data$lrn
    tsk = data$tsk
    tsk_train = subsetTask(tsk, instance$train)
    tsk_test = subsetTask(tsk, instance$test)
    rsmp_tuning = data$rsmp_tuning
    ps = data$ps
    ctrl = data$ctrl
    #parallelStartMulticore(cpus = rsmp_tuning$iters %??% 4, level = "mlr.resample") #does not work :(
    tune_res = tuneParamsMultiCrit(learner = lrn, task = tsk_train, resampling = rsmp_tuning, measures = list(tpr, fpr), par.set = ps, control = ctrl)
    #parallelStop()
    y_outer = lapply(tune_res$x, function(x) {
      lrn2 = setHyperPars(lrn, par.vals = x)
      mod = train(lrn2, tsk_train)
      pred = predict(mod, task = tsk_test)
      performance(pred, measures = list(tpr, fpr))
    })
    y_outer = do.call(rbind, y_outer)
    return(list(tune_res = tune_res, outer_res = y_outer))
  }
  
  addAlgorithm("mlr", fun = algo_mlr)
  
  addExperiments(repls = rsmp_outer$iters)
  submitJobs(resources = list(ncpus = 1))
  waitForJobs()
  res = reduceResultsList(findDone(), fun = function(res, job) {
    list(
      x = res$tune_res$x,
      inner_y = res$tune_res$y,
      outer_y = res$outer_res
    )
  })
  saveRDS(res, "benchmark_res.rds")
}



