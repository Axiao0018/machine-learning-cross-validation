install.packages(c("tidyverse","xgboost","lightgbm","randomForest","glmnet","gbm","e1071","PRROC","caret"))
################################################################################
# 14 个产地 One‑vs‑Rest 二分类机器学习基准测试
# 数据概况：
#   42 个样本；14 个 group，每组 3 个生物学重复；1190 个代谢物特征
# 任务：
#   target_group = 1:14
#   当前目标group → status = 1；其余group → status = 0
#   每个OvR任务：正样本=3，负样本=39
# 对比算法(10种)：
#   1.L2‑LR  2.Elastic‑Net  3.Naive‑Bayes  4.KNN  5.SVM‑Linear
#   6.SVM‑RBF  7.Random‑Forest  8.GBM  9.LightGBM  10.XGBoost
# 验证策略：严格留一法交叉验证 LOOCV
# 调参目标：最大化 LOOCV PR‑AUC
# 输出指标：
#   Train_PR_AUC（完整训练集PR‑AUC）
#   LOOCV_PR_AUC（留一法PR‑AUC）
#   LOOCV_ROC_AUC（留一法ROC‑AUC）
#   Overfitting_Gap = Train_PR_AUC − LOOCV_PR_AUC（过拟合差值，越大过拟合越重）
# 重要约束：
#   本脚本仅做算法基准benchmark；不运行SHAP，不筛选biomarker，不训练下游分析的最终模型
#   ✅预处理在每个fold内部完成，杜绝测试集数据泄露
################################################################################

# --------------------------- 0.加载依赖包 ------------------------------------
library(tidyverse)
library(xgboost)
library(lightgbm)
library(randomForest)
library(glmnet)
library(gbm)
library(e1071)
library(PRROC)
library(caret)

# --------------------------- 1.全局基础设置 -----------------------------------
setwd("C:\\Users\\86135\\Desktop\\moxing")
input_file  <- "data.csv"   # 修改为你的csv路径
output_file <- "model_performance_summary_10algorithms.csv"
set.seed(123)                  # 全局随机种子，保证可复现

# --------------------------- 2.读取输入数据 ------------------------------------
data_df <- read.csv(
  input_file,
  header = TRUE,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

cat("\n================ 数据信息 ================\n")
cat("样本数:", nrow(data_df), "\n")
cat("总列数:", ncol(data_df), "\n")
cat("代谢物特征数:", ncol(data_df) - 2, "\n")
cat("group 数量:", length(unique(data_df$group)), "\n")

# --------------------------- 3.数据合法性校验 ----------------------------------
if (!all(c("ID", "group") %in% colnames(data_df))) {
  stop("【错误】输入csv必须包含ID、group两列！")
}
if (nrow(data_df) != 42) {
  warning("【警告】当前样本数不等于42，请核对输入数据！")
}
if (length(unique(data_df$group)) != 14) {
  stop("【错误】group分组数量不等于14，请核对group列！")
}

group_table <- table(data_df$group)
print(group_table)
if (any(group_table != 3)) {
  stop("【错误】不是每个group都恰好3个生物学重复，请核对group列！")
}

# 提取代谢物特征矩阵，强制转为数值矩阵
X_raw <- data_df[, 3:ncol(data_df), drop = FALSE]
X_raw <- as.matrix(data.frame(lapply(X_raw, as.numeric), check.names = FALSE))
storage.mode(X_raw) <- "double"

if (anyNA(X_raw)) {
  stop("【错误】代谢物矩阵存在NA缺失值！运行前请先处理缺失。")
}
cat("\n✅ 数据结构检查全部通过\n")

# --------------------------- 4. PR‑AUC / ROC‑AUC 计算函数 ----------------------
calculate_pr_auc <- function(y, prob) {
  y <- as.numeric(y)
  prob <- as.numeric(prob)
  if (any(!is.finite(prob))) return(NA_real_)
  if (length(unique(y)) < 2) return(NA_real_)
  res <- PRROC::pr.curve(scores.class0 = prob[y == 1],
                         scores.class1 = prob[y == 0],
                         curve = FALSE)
  return(as.numeric(res$auc.integral))
}

calculate_roc_auc <- function(y, prob) {
  y <- as.numeric(y)
  prob <- as.numeric(prob)
  if (any(!is.finite(prob))) return(NA_real_)
  if (length(unique(y)) < 2) return(NA_real_)
  res <- PRROC::roc.curve(scores.class0 = prob[y == 1],
                          scores.class1 = prob[y == 0],
                          curve = FALSE)
  return(as.numeric(res$auc))
}

# ---------------------- 5.单fold内部预处理（关键！防止数据泄露） -----------------
## 使用fold内训练集计算均值、标准差；过滤零方差特征；再变换训练集+测试集
preprocess_fold <- function(X_train, X_test) {
  feature_sd <- apply(X_train, 2, sd)
  keep <- which(is.finite(feature_sd) & feature_sd > 0)
  if (length(keep) == 0) stop("【错误】本fold无有效特征！")
  
  X_train <- X_train[, keep, drop = FALSE]
  X_test  <- X_test[, keep, drop = FALSE]
  
  mu    <- colMeans(X_train)
  sigma <- apply(X_train, 2, sd)
  
  X_train_scaled <- sweep(X_train, 2, mu, "-")
  X_train_scaled <- sweep(X_train_scaled, 2, sigma, "/")
  
  X_test_scaled  <- sweep(X_test,  2, mu, "-")
  X_test_scaled  <- sweep(X_test_scaled, 2, sigma, "/")
  
  return(list(train = X_train_scaled, test = X_test_scaled))
}

# ----------------------- 6.全数据集预处理（用于计算训练集性能） ----------------
preprocess_full <- function(X) {
  feature_sd <- apply(X, 2, sd)
  keep <- which(is.finite(feature_sd) & feature_sd > 0)
  X2 <- X[, keep, drop = FALSE]
  mu    <- colMeans(X2)
  sigma <- apply(X2, 2, sd)
  X_scaled <- sweep(X2, 2, mu, "-")
  X_scaled <- sweep(X_scaled, 2, sigma, "/")
  return(X_scaled)
}

# ----------------------- 7.模型训练+预测概率通用函数 ---------------------------
fit_predict_model <- function(algorithm, X_train, y_train, X_test, params) {
  y_factor <- factor(y_train, levels = c(0, 1))
  
  if (algorithm == "L2‑LR") {
    model <- glmnet::glmnet(x = X_train, y = y_train, family = "binomial",
                            alpha = 0, lambda = params$lambda, standardize = FALSE)
    prob <- as.numeric(predict(model, newx = X_test, s = params$lambda, type = "response"))
    
  } else if (algorithm == "Elastic‑Net") {
    model <- glmnet::glmnet(x = X_train, y = y_train, family = "binomial",
                            alpha = params$alpha, lambda = params$lambda, standardize = FALSE)
    prob <- as.numeric(predict(model, newx = X_test, s = params$lambda, type = "response"))
    
  } else if (algorithm == "Naive‑Bayes") {
    train_df <- as.data.frame(X_train)
    train_df$status <- y_factor
    test_df  <- as.data.frame(X_test)
    model <- e1071::naiveBayes(status ~ ., data = train_df, laplace = 1)
    pred <- predict(model, newdata = test_df, type = "raw")
    prob <- as.numeric(pred[, "1"])
    
  } else if (algorithm == "KNN") {
    model <- caret::knn3(x = X_train, y = y_factor, k = params$k)
    pred <- predict(model, newdata = X_test, type = "prob")
    prob <- as.numeric(pred[, "1"])
    
  } else if (algorithm == "SVM‑Linear") {
    model <- e1071::svm(x = X_train, y = y_factor, kernel = "linear",
                        cost = params$cost, probability = TRUE, scale = FALSE)
    pred <- predict(model, X_test, probability = TRUE)
    prob_matrix <- attr(pred, "probabilities")
    prob <- as.numeric(prob_matrix[, "1"])
    
  } else if (algorithm == "SVM‑RBF") {
    model <- e1071::svm(x = X_train, y = y_factor, kernel = "radial",
                        cost = params$cost, gamma = params$gamma, probability = TRUE, scale = FALSE)
    pred <- predict(model, X_test, probability = TRUE)
    prob_matrix <- attr(pred, "probabilities")
    prob <- as.numeric(prob_matrix[, "1"])
    
  } else if (algorithm == "Random‑Forest") {
    model <- randomForest::randomForest(x = X_train, y = y_factor,
                                        ntree = params$ntree,
                                        mtry = min(params$mtry, ncol(X_train)))
    pred <- predict(model, X_test, type = "prob")
    prob <- as.numeric(pred[, "1"])
    
  } else if (algorithm == "GBM") {
    train_df <- as.data.frame(X_train)
    train_df$status <- y_train
    model <- gbm::gbm(formula = status ~ ., data = train_df, distribution = "bernoulli",
                      n.trees = params$n.trees, interaction.depth = params$interaction.depth,
                      shrinkage = params$shrinkage, n.minobsinnode = 1,
                      bag.fraction = 1, train.fraction = 1, verbose = FALSE)
    prob <- as.numeric(predict(model, newdata = as.data.frame(X_test),
                               n.trees = params$n.trees, type = "response"))
    
  } else if (algorithm == "LightGBM") {
    dtrain <- lightgbm::lgb.Dataset(data = X_train, label = y_train)
    lgb_params <- list(
      objective = "binary", metric = "binary_logloss",
      learning_rate = params$learning_rate, max_depth = params$max_depth,
      min_data_in_leaf = 1, feature_pre_filter = FALSE, verbosity = -1, num_threads = 1
    )
    model <- lightgbm::lgb.train(params = lgb_params, data = dtrain,
                                 nrounds = params$num_iterations, verbose = -1)
    prob <- as.numeric(predict(model, X_test))
    
  } else if (algorithm == "XGBoost") {
    dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train)
    dtest  <- xgboost::xgb.DMatrix(data = X_test)
    xgb_params <- list(
      objective = "binary:logistic", eval_metric = "logloss",
      max_depth = params$max_depth, eta = params$learning_rate,
      min_child_weight = 1, subsample = 1, colsample_bytree = 1, nthread = 1
    )
    model <- xgboost::xgb.train(params = xgb_params, data = dtrain,
                                nrounds = params$n_estimators, verbose = 0)
    prob <- as.numeric(predict(model, dtest))
    
  } else {
    stop(paste("未知算法名称：", algorithm))
  }
  prob <- pmin(pmax(prob, 0), 1)
  return(prob)
}

# ------------------------ 8.一套超参数运行完整LOOCV ----------------------------
run_loocv <- function(X, y, algorithm, params) {
  n <- nrow(X)
  oof_prob <- rep(NA_real_, n)
  
  for (i in seq_len(n)) {
    train_idx <- setdiff(seq_len(n), i)
    test_idx  <- i
    X_train_raw <- X[train_idx, , drop = FALSE]
    X_test_raw  <- X[test_idx, , drop = FALSE]
    y_train     <- y[train_idx]
    
    pp <- preprocess_fold(X_train_raw, X_test_raw)
    X_train <- pp$train
    X_test  <- pp$test
    
    set.seed(10000 + i)
    prob <- tryCatch({
      fit_predict_model(algorithm, X_train, y_train, X_test, params)
    }, error = function(e) {
      message("LOOCV fold ", i, "报错：", conditionMessage(e))
      return(NA_real_)
    })
    oof_prob[i] <- prob[1]
  }
  
  if (anyNA(oof_prob)) {
    return(list(pr_auc = NA_real_, roc_auc = NA_real_, prob = oof_prob))
  }
  pr_auc  <- calculate_pr_auc(y, oof_prob)
  roc_auc <- calculate_roc_auc(y, oof_prob)
  return(list(pr_auc = pr_auc, roc_auc = roc_auc, prob = oof_prob))
}

# ------------------------ 9.网格超参数搜索函数 ---------------------------------
tune_algorithm <- function(X, y, algorithm) {
  p <- ncol(X)
  if (algorithm == "L2‑LR") {
    grid <- expand.grid(lambda = c(0.001, 0.01, 0.1, 1, 10))
  } else if (algorithm == "Elastic‑Net") {
    grid <- expand.grid(alpha = c(0.1,0.5,0.9), lambda = c(0.001,0.01,0.1,1,10))
  } else if (algorithm == "Naive‑Bayes") {
    grid <- data.frame(dummy = 1)
  } else if (algorithm == "KNN") {
    grid <- expand.grid(k = c(1,3,5,7,9))
  } else if (algorithm == "SVM‑Linear") {
    grid <- expand.grid(cost = c(0.01,0.1,1,10,100))
  } else if (algorithm == "SVM‑RBF") {
    grid <- expand.grid(cost = c(0.1,1,10), gamma = c(1/p, 0.01, 0.001))
  } else if (algorithm == "Random‑Forest") {
    grid <- expand.grid(ntree = c(300,500),
                        mtry = unique(pmax(1, round(c(sqrt(p), p/10)))))
  } else if (algorithm == "GBM") {
    grid <- expand.grid(n.trees = c(100,300), interaction.depth = c(1,3),
                        shrinkage = c(0.01,0.05))
  } else if (algorithm == "LightGBM") {
    grid <- expand.grid(max_depth = c(2,3,5), learning_rate = c(0.01,0.05),
                        num_iterations = c(100,300))
  } else if (algorithm == "XGBoost") {
    grid <- expand.grid(max_depth = c(2,3,5), learning_rate = c(0.01,0.05),
                        n_estimators = c(100,300))
  } else {
    stop(paste("未识别算法：", algorithm))
  }
  
  best_pr  <- -Inf
  best_roc <- NA_real_
  best_params <- NULL
  best_prob   <- NULL
  
  for (g in seq_len(nrow(grid))) {
    params <- as.list(grid[g, , drop = FALSE])
    if (algorithm == "Naive‑Bayes") params <- list()
    
    res <- run_loocv(X = X, y = y, algorithm = algorithm, params = params)
    cur_pr <- res$pr_auc
    
    if (!is.na(cur_pr) && cur_pr > best_pr) {
      best_pr  <- cur_pr
      best_roc <- res$roc_auc
      best_params <- params
      best_prob   <- res$prob
    }
  }
  
  if (is.infinite(best_pr)) {
    return(list(pr_auc = NA_real_, roc_auc = NA_real_, params = NULL, prob = NULL))
  }
  return(list(pr_auc = best_pr, roc_auc = best_roc, params = best_params, prob = best_prob))
}

# --------------------10.计算完整数据集训练集PR‑AUC（用于算gap）----------------
calculate_training_pr <- function(X, y, algorithm, params) {
  X_scaled <- preprocess_full(X)
  set.seed(12345)
  train_prob <- tryCatch({
    fit_predict_model(algorithm, X_train = X_scaled, y_train = y,
                      X_test = X_scaled, params = params)
  }, error = function(e) {
    message("完整集训练报错：", conditionMessage(e))
    return(rep(NA_real_, length(y)))
  })
  if (anyNA(train_prob)) return(NA_real_)
  return(calculate_pr_auc(y, train_prob))
}

# ----------------------11.定义全部待运行算法列表 -------------------------------
algorithms <- c("L2‑LR", "Elastic‑Net", "Naive‑Bayes", "KNN",
                "SVM‑Linear", "SVM‑RBF", "Random‑Forest",
                "GBM", "LightGBM", "XGBoost")

# ----------------------12.初始化结果数据框 -------------------------------------
performance_results <- data.frame(
  target_group = integer(),
  algorithm    = character(),
  Train_PR_AUC = numeric(),
  LOOCV_PR_AUC = numeric(),
  LOOCV_ROC_AUC= numeric(),
  Overfitting_Gap = numeric(),
  stringsAsFactors = FALSE
)

# ----------------------13.主循环：14个OvR任务 ----------------------------------
for (target_group in 1:14) {
  cat("\n\n", "====================================================\n",
      ">>>>> 开始OvR任务 target_group = ", target_group, "\n",
      "====================================================\n", sep = "")
  
  # 动态生成0‑1标签
  status <- ifelse(data_df$group == target_group, 1, 0)
  cat("正样本数：", sum(status == 1), " | 负样本数：", sum(status == 0), "\n")
  
  if (sum(status ==1) !=3 || sum(status ==0)!=39) {
    stop(paste0("target_group = ",target_group,"样本数量不是3 vs 39，请检查group列！"))
  }
  
  # 遍历10种算法
  for (algorithm in algorithms) {
    cat("\n------------------------------------\n")
    cat("target_group:", target_group, "  算法：", algorithm, "\n")
    
    tuned_res <- tune_algorithm(X = X_raw, y = status, algorithm = algorithm)
    loocv_pr  <- tuned_res$pr_auc
    loocv_roc <- tuned_res$roc_auc
    
    if (is.na(loocv_pr) || is.null(tuned_res$params)) {
      train_pr <- NA_real_
      gap      <- NA_real_
    } else {
      train_pr <- calculate_training_pr(X = X_raw, y = status,
                                        algorithm = algorithm, params = tuned_res$params)
      gap <- train_pr - loocv_pr
    }
    
    # 追加一行结果
    new_row <- data.frame(
      target_group = target_group,
      algorithm    = algorithm,
      Train_PR_AUC = train_pr,
      LOOCV_PR_AUC = loocv_pr,
      LOOCV_ROC_AUC= loocv_roc,
      Overfitting_Gap = gap,
      stringsAsFactors = FALSE
    )
    performance_results <- bind_rows(performance_results, new_row)
    
    # 每跑完一个算法就写盘，防止程序崩溃丢失全部结果
    write.csv(performance_results, output_file, row.names = FALSE)
    
    cat(sprintf("Train PR‑AUC: %.4f | LOOCV PR‑AUC: %.4f | LOOCV ROC‑AUC: %.4f | Gap: %.4f\n",
                train_pr, loocv_pr, loocv_roc, gap))
    cat("最优超参数：\n")
    print(tuned_res$params)
    gc() #内存回收
  }
}

# -----------------------14.结果整理与输出 --------------------------------------
performance_results <- performance_results %>% arrange(target_group, algorithm)
write.csv(performance_results, output_file, row.names = FALSE)

cat("\n\n====================================================\n")
cat("✅全部14×10 OvR基准测试运行完成！\n")
cat("结果输出文件：", output_file, "\n\n")
print(performance_results)

# 输出描述性汇总：各算法跨14个任务平均指标
algorithm_summary <- performance_results %>%
  group_by(algorithm) %>%
  summarise(
    Mean_Train_PR_AUC     = mean(Train_PR_AUC, na.rm = TRUE),
    Mean_LOOCV_PR_AUC     = mean(LOOCV_PR_AUC, na.rm = TRUE),
    Mean_LOOCV_ROC_AUC    = mean(LOOCV_ROC_AUC, na.rm = TRUE),
    Mean_Overfitting_Gap  = mean(Overfitting_Gap, na.rm = TRUE),
    .groups = "drop"
  )

cat("\n================ 各算法跨全部OvR任务平均表现 ================\n")
print(algorithm_summary)

