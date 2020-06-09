library(mlr)
library(mlrMBO)
library(checkmate)
library(mlr3misc)
library(data.table)
library(ggplot2)
library(parallelMap)
set.seed(1)
tsk = spam.task
#tsk = sonar.task
n_evals = 100
theme_set(theme_bw())

res_file = "example_parego_res.rds"

ps = makeParamSet(
  makeNumericParam("cost", lower = -15, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("gamma", lower = -15, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("thresh", lower = 0, upper = 1)
)
x = sampleValue(ps, trafo = TRUE)

rsmp_tuning = makeResampleDesc("CV", iters = 5)
rsmp_outer = makeResampleDesc("Holdout")
rsmp_outer = makeResampleInstance(rsmp_outer, tsk)

tsk_train = subsetTask(tsk, rsmp_outer$train.inds[[1]])
tsk_test = subsetTask(tsk, rsmp_outer$test.inds[[1]])

data = list(
  lrn = makeLearner("classif.svm", predict.type = "prob") , 
  tsk = tsk_train, 
  rsmp_tuning = makeResampleInstance(rsmp_tuning, tsk_train)
)
# data = list(
#   lrn = makeLearner("classif.xgboost", predict.type = "prob") 
# )

measures = list(tpr, fpr)

get_tpr_fpr_thr = function(pred, task) {
  probs = sort(unique(getPredictionProbabilities(pred, cl = task$task.desc$negative)))
  tprfprs = map_dtr(probs, function(p) {
    pred_thr = setThreshold(pred, p)
    as.data.frame(t(performance(pred_thr, measures = list(tpr, fpr))))
  })
  tprfprs = unique(tprfprs)
}

obj = makeMultiObjectiveFunction("mlr", fn = function(x, outer_split = 1, data) {
  lrn_x = x[intersect(names(x), getParamIds(getParamSet(data$lrn)))]
  res = resample(
    learner = setHyperPars(data$lrn, par.vals = lrn_x), 
    task = data$tsk, 
    resampling = data$rsmp_tuning, 
    measures = list(tpr, fpr)
  )
  tprfprs = get_tpr_fpr_thr(res$pred, data$tsk)
  #tres = tuneThreshold(pred = res$pred)
  pred_thr = setThreshold(res$pred, x$thresh)
  y = performance(pred_thr, measures = list(tpr, fpr))
  attr(y, "extras") = list(".curve" = tprfprs)
  # plot(tprfprs)
  # points(y["tpr"], y["fpr"], col = "red")
  return(y)
}, has.simple.signature = FALSE, par.set = ps, noisy = FALSE, minimize = map_lgl(measures, "minimize"), n.objectives = length(measures))

# MBO
mbo_ctrl = makeMBOControl(n.objectives = 2, y.name = map_chr(measures, "id"))
mbo_ctrl = setMBOControlTermination(mbo_ctrl, max.evals = n_evals, iters = n_evals)
mbo_ctrl = setMBOControlMultiObj(mbo_ctrl, method = "parego")

#des = generateDesign(n = 4L * sum(getParamLengths(ps)), par.set = ps)
#xs = dfRowsToList(des, par.set = ps)
#xs = lapply(xs, trafoValue, par = ps)
#ys = lapply(xs, obj, data = data)
#des_y = cbind(des, as.data.frame(do.call(rbind, ys)))

if (!file.exists(res_file)) {
  # run MBO
  set.seed(1)
  parallelStartMulticore(4, level = "mlr.resample")
  res_mbo = mbo(obj, control = mbo_ctrl, more.args = list(data = data))  
  
  # run RS
  res_rs = {
    set.seed(1)
    des_random = generateRandomDesign(n = mbo_ctrl$max.evals, par.set = getParamSet(obj))
    xs = dfRowsToList(des_random, par.set = getParamSet(obj))
    xs = lapply(xs, trafoValue, par = getParamSet(obj))
    res_random = parallelLapply(xs = xs, fun = obj, data = data)
    res_rs = do.call(rbind, res_random)
    res_rs = as.data.table(res_rs)
    res_rs$dominated = emoa::is_dominated(t(as.matrix(res_rs[, .(fpr,1-tpr)])))
    res_rs$extras = lapply(res_random, attr, "extras")
    res_rs$xs = xs
    res_rs = cbind(res_rs, des_random)
  }
  
  eval_x = function(x, trafo = TRUE) {
    if (trafo) {
      x = trafoValue(par = ps, x = x)  
    }
    lrn_x = x[intersect(names(x), getParamIds(getParamSet(data$lrn)))]
    lrn2 = setHyperPars(data$lrn, par.vals = lrn_x)
    mod = train(lrn2, tsk_train)
    pred = predict(mod, task = tsk_test)
    tprfprs = get_tpr_fpr_thr(pred, data$tsk)
    
    #tres = tuneThreshold(pred = pred)
    pred_thr = setThreshold(pred, x$thresh)
    y = performance(pred_thr, measures = list(tpr, fpr))
    y = as.data.table(t(y))
    y[, .curve := list(tprfprs)]
    return(y)
  }
  # calc mbo performance on testv
  y_outer_mbo = map_dtr(res_mbo$pareto.set, eval_x)
  
  # calc rs performance on test
  y_outer_rs = map_dtr(res_rs[dominated == FALSE,]$xs, eval_x, trafo = FALSE)
  
  saveRDS(list(res_mbo = res_mbo, y_outer_mbo = y_outer_mbo, res_rs = res_rs, y_outer_rs = y_outer_rs), res_file)
} else {
  tmp = readRDS(res_file)
  res_mbo = tmp$res_mbo
  y_outer_mbo = tmp$y_outer_mbo
  y_outer_rs = tmp$y_outer_rs
  res_rs = tmp$res_rs
}

res_rs[, tuner := "rs"]
opdf = as.data.frame(res_mbo$opt.path)
pareto_df = unique(as.data.frame(res_mbo$pareto.front))
pareto_df$dominated = FALSE
opdf = merge(opdf, pareto_df, by = colnames(res_mbo$pareto.front), all.x = TRUE)
setDT(opdf)
opdf$tuner = "parEGO"
opdf[is.na(dominated), dominated := TRUE]
opdf = rbindlist(list(opdf, res_rs[tuner == "rs"]), fill = TRUE)

#tuning performance
g = ggplot(opdf[dominated == FALSE,], aes(x = fpr, y = tpr, color = tuner))
#g = g + geom_point(data = subdf_inner, alpha = 0.1, size = 1)
#g = g + geom_point(data = subdf_outer, alpha = 0.1, size = 1)
g = g + geom_point(data = opdf, alpha = 0.5)
g = g + geom_step()
# g = g + geom_point(data = y_outer_mbo, size = 2)
g = g + coord_cartesian(ylim = c(0.5,1), xlim = c(0,0.5))
g = g + labs(
  title = "Tuning: SVM (cost, gama, threshold) on spam dataset", subtitle = "positive = nonspam")
if (interactive()) {
  print(g)
}
ggsave("../images/example_parego_spam.png", g, height = 5, width = 6)

emoa::dominated_hypervolume(points = t(as.matrix(opdf[tuner == "rs", .(fpr,1-tpr)])), ref = c(1,1))
# 0.9586348
emoa::dominated_hypervolume(points = t(as.matrix(opdf[tuner == "parEGO", .(fpr,1-tpr)])), ref = c(1,1))
# 0.9651275

#outer performance
y_outer_mbo$tuner = "parEGO"
y_outer_rs$tuner = "rs"
y_outer_mbo$type = "validation"
y_outer_rs$type = "validation"
opdf$type = "tuning"
g = ggplot(opdf[dominated == FALSE,], aes(x = fpr, y = tpr, color = tuner, shape = type))
g = g + geom_point(alpha = 0.5)
g = g + geom_step(alpha = 0.5)
g = g + geom_point(data = rbind(y_outer_mbo, y_outer_rs), size = 2)
g = g + coord_cartesian(ylim = c(0.5,1), xlim = c(0,0.5))
g = g + labs(title = "Tuning Validation: SVM on spam dataset", subtitle = "positive = nonspam")
if (interactive()) {
  print(g)
}
ggsave("../images/example_parego_spam_outer.png", g, height = 5, width = 6)

# outer pareto

emoa::dominated_hypervolume(points = t(as.matrix(y_outer_rs[, .(fpr,1-tpr)])), ref = c(1,1))
# 0.961211
emoa::dominated_hypervolume(points = t(as.matrix(y_outer_mbo[, .(fpr,1-tpr)])), ref = c(1,1))
# 0.9601199
y_outer_rs$dominated = emoa::is_dominated(points = t(as.matrix(y_outer_rs[, .(fpr,1-tpr)])))
y_outer_mbo$dominated = emoa::is_dominated(points = t(as.matrix(y_outer_mbo[, .(fpr,1-tpr)])))
tmp = rbind(y_outer_mbo, y_outer_rs)
g = ggplot(data = tmp[dominated == FALSE,], mapping = aes(x = fpr, y = tpr, color = tuner, shape = type))
g = g + geom_point(data = tmp, alpha = 0.5)
g = g + geom_step()
g = g + coord_cartesian(ylim = c(0.5,1), xlim = c(0,0.5))
g = g + labs(title = "Tuning Validation: SVM on spam dataset", subtitle = "positive = nonspam")
if (interactive()) {
  print(g)
}
ggsave("../images/example_parego_spam_outer_pareto.png", g, height = 5, width = 6)



subdf_inner = map_dtr(res_mbo$opt.path$env$extra, ".curve")
subdf_inner$type = "inner threshold variation"
subdf_outer = rbindlist(y_outer_mbo$.curve)
subdf_outer$type = "outer threshold variation"
y_outer_mbo$type = "pareto vaildation"
