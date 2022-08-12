# Correlation coefficient custom performance metric for tidymodels.
# https://www.tidymodels.org/learn/develop/metrics/
# Vector version.
spearman_cor_vec = function(truth, estimate, na_rm = TRUE) {
  
  spearman_cor_impl = function(truth, estimate) {
    cor(truth, estimate, method = "spearman")
  }
  
  metric_vec_template(
    metric_impl = spearman_cor_impl,
    truth = truth, 
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric"
  )
}
# Data frame version. 
# https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/UseMethod
spearman_cor = function(data) {
  UseMethod("spearman_cor")
}

spearman_cor = new_numeric_metric(spearman_cor, direction = "maximize")

spearman_cor.data.frame = function(data, truth, estimate, na_rm = TRUE) {
  
  data_grouped = data %>%
    group_by(Sample, )
  
  metric_summarizer(
    metric_nm = "spearman_cor",
    metric_fn = spearman_cor_vec,
    data = data_grouped,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    na_rm = na_rm
  )
  
}

## Test stuff.
#ccc_with_bias <- function(data, truth, estimate, na_rm = TRUE, ...) {
#  ccc(
#    data = data,
#    truth = !! rlang::enquo(truth),
#    estimate = !! rlang::enquo(estimate),
#    # set bias = TRUE
#    bias = TRUE,
#    na_rm = na_rm,
#    ...
# )
#}
#
## Use `new_numeric_metric()` to formalize this new metric function
#ccc_with_bias <- new_numeric_metric(ccc_with_bias, "maximize")

mse_vec <- function(truth, estimate, na_rm = TRUE, ...) {
  
  mse_impl <- function(truth, estimate) {
    mean((truth - estimate) ^ 2)
  }
  
  metric_vec_template(
    metric_impl = mse_impl,
    truth = truth, 
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
  
}

mse <- function(data, ...) {
  UseMethod("mse")
}

mse <- new_numeric_metric(mse, direction = "minimize")

mse.data.frame <- function(data, truth, estimate, na_rm = TRUE, ...) {
  
  data_grouped = data %>%
    group_by(Sample)
  
  metric_summarizer(
    metric_nm = "mse",
    metric_fn = mse_vec,
    data = data_grouped,
    truth = !! enquo(truth),
    estimate = !! enquo(estimate), 
    na_rm = na_rm,
    ...
  )
  
}
