# Library imports
library(MatchIt)
library(fixest)
library(stargazer)
library(modelsummary)
library(arrow)
library(MASS)
library(e1071)

# Data import
panel = read_parquet('../data/working_data.parquet')

# Box-Cox transformation
hist(panel$age, main = "Original Data", col = "lightblue")
qqnorm(panel$age, main = "Q-Q Plot (Original)")
qqline(panel$age, col = "red")

boxcox_result <- boxcox(lm(panel$age ~ 1), lambda = seq(-2, 2, 0.1))

best_lambda <- boxcox_result$x[which.max(boxcox_result$y)]
print(paste("Optimal lambda:", round(best_lambda, 3)))

transformed_data <- (panel$age^best_lambda - 1) / best_lambda

hist(transformed_data, main = "Transformed (λ = 0.263)", col = "lightgreen")
qqnorm(transformed_data, main = "Q-Q Plot (Transformed)")
qqline(transformed_data, col = "blue")
boxplot(transformed_data, main = "Boxplot (Transformed)", col = "lightgreen")

panel$age_box_cox = (panel$age^best_lambda - 1) / best_lambda

# PSM
psm_nn <- matchit(has_disablity ~ age_box_cox
                 + educ_level_university + is_female + 
                 is_married + harmfull_job + is_employed,
                  data = panel ,
                  s.weights = ~ inwgt
               )
print(summary(psm_nn, un = FALSE))

psm_full_logit <- matchit(has_disablity ~ age_box_cox
                 + educ_level_university + is_female + 
                 is_married + harmfull_job + is_employed,
                  data = panel ,
                  s.weights = ~ inwgt,
                  method = "full",
                  link = "logit",
                  distance = "glm",
                  #exact = ~log_age
               )
print(summary(psm_full_logit, un = FALSE))

# Compare PSM algorithms
love.plot(psm_full_probit, stats = c("m", "ks"), poly = 2, abs = TRUE,
          weights = list(nn = psm_nn),
          drop.distance = TRUE, thresholds = c(m = .1),
          var.order = "unadjusted", binary = "std",
          shapes = c("circle filled", "triangle", "square"), 
          colors = c("lightcoral", "steelblue", "lightblue"),
          sample.names = c("Original", "Full Matching", "NN Matching"),
          position = "bottom")

panel_matched <- match_data(psm_full_logit)
panel_matched