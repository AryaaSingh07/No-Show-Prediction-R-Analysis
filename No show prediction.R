# =============================================================================
# HEALTHCARE NO-SHOW PREDICTION
# Predictive Analytics — Portfolio Project
# Dataset: Medical Appointment No-Shows in Brazil (Kaggle, 110,527 records)
# Author: Aryaa Singh | UT Dallas — MS Business Analytics & AI
# =============================================================================
# Dataset download:
# https://www.kaggle.com/datasets/joniarroba/noshowappointments
# Save as: KaggleV2-May-2016.csv in your working directory
# =============================================================================


# =============================================================================
# SECTION 0 — SETUP & LIBRARIES
# =============================================================================

# Install packages if not already installed
packages_needed <- c(
  "tidyverse",   # data manipulation + ggplot2
  "caret",       # model training + confusion matrix
  "pROC",        # ROC curves + AUC
  "corrplot",    # correlation matrix visualization
  "gridExtra",   # multi-panel plots
  "scales",      # axis formatting
  "lubridate",   # date handling
  "knitr"        # clean table output
)

for (pkg in packages_needed) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

set.seed(42)   # reproducibility

cat("=== HEALTHCARE NO-SHOW PREDICTION ANALYSIS ===\n")
cat("Author: Aryaa Singh | UT Dallas\n\n")


# =============================================================================
# SECTION 1 — DATA LOADING & FIRST LOOK
# =============================================================================

cat("--- SECTION 1: Loading Data ---\n")

# Load dataset
df_raw <- read.csv("/Users/aryaasingh/Desktop/Healthcare Project/Dataset/KaggleV2-May-2016.csv", 
                   stringsAsFactors = FALSE)

setwd("/Users/aryaasingh/Desktop/Healthcare Project")

cat("Dimensions:", nrow(df_raw), "rows x", ncol(df_raw), "columns\n")
cat("\nColumn names:\n")
print(names(df_raw))

cat("\nFirst 6 rows:\n")
print(head(df_raw))

cat("\nData types:\n")
print(str(df_raw))


# =============================================================================
# SECTION 2 — DATA CLEANING & FEATURE ENGINEERING
# =============================================================================

cat("\n--- SECTION 2: Data Cleaning & Feature Engineering ---\n")

df <- df_raw

# ── 2.1 Rename columns for clarity ──────────────────────────────────────────
df <- df %>%
  rename(
    patient_id       = PatientId,
    appointment_id   = AppointmentID,
    gender           = Gender,
    scheduled_day    = ScheduledDay,
    appointment_day  = AppointmentDay,
    age              = Age,
    neighbourhood    = Neighbourhood,
    scholarship      = Scholarship,
    hypertension     = Hipertension,
    diabetes         = Diabetes,
    alcoholism       = Alcoholism,
    handicap         = Handcap,
    sms_received     = SMS_received,
    no_show          = No.show
  )

# ── 2.2 Convert target variable: 1 = No-Show, 0 = Showed Up ─────────────────
df$no_show_binary <- ifelse(df$no_show == "Yes", 1, 0)

cat("No-show rate:\n")
print(prop.table(table(df$no_show_binary)))
# Expected: ~22.8% no-shows

# ── 2.3 Parse dates ──────────────────────────────────────────────────────────
df$scheduled_day    <- as.Date(df$scheduled_day)
df$appointment_day  <- as.Date(df$appointment_day)

# ── 2.4 Engineer: lead_time (days between scheduling and appointment) ─────────
df$lead_time <- as.numeric(df$appointment_day - df$scheduled_day)

# Remove impossible values (negative lead time = data entry error)
cat("Rows with negative lead time (removed):", sum(df$lead_time < 0), "\n")
df <- df %>% filter(lead_time >= 0)

# ── 2.5 Engineer: appointment day of week ────────────────────────────────────
df$appt_weekday     <- weekdays(df$appointment_day)
df$appt_weekday_num <- as.integer(format(df$appointment_day, "%u"))  # 1=Mon, 7=Sun

# ── 2.6 Engineer: age groups ─────────────────────────────────────────────────
# Remove impossible ages
cat("Rows with age < 0 (removed):", sum(df$age < 0), "\n")
df <- df %>% filter(age >= 0)

df$age_group <- cut(df$age,
                    breaks = c(-1, 12, 18, 35, 60, 120),
                    labels = c("Child(0-12)", "Teen(13-18)", "YoungAdult(19-35)",
                               "MiddleAge(36-60)", "Senior(61+)")
)

# ── 2.7 Engineer: prior_noshow_count per patient ─────────────────────────────
# Count historical no-shows per patient (using appointment order)
df <- df %>%
  arrange(patient_id, scheduled_day) %>%
  group_by(patient_id) %>%
  mutate(prior_noshow_count = cumsum(lag(no_show_binary, default = 0))) %>%
  ungroup()

# ── 2.8 Engineer: high_risk flag (lead_time > 5 days AND prior no-shows > 1) ─
df$high_risk <- as.integer(df$lead_time > 5 & df$prior_noshow_count > 1)

# ── 2.9 Final cleaning ────────────────────────────────────────────────────────
# Ensure binary columns are integer
binary_cols <- c("scholarship","hypertension","diabetes","alcoholism",
                 "sms_received","no_show_binary","high_risk")
df[binary_cols] <- lapply(df[binary_cols], as.integer)

# Clamp handicap to binary (original has values 0-4, likely encoding issue)
df$handicap <- as.integer(df$handicap > 0)

cat("\nCleaned dataset dimensions:", nrow(df), "rows x", ncol(df), "cols\n")
cat("Missing values per column:\n")
print(colSums(is.na(df)))


# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

cat("\n--- SECTION 3: Exploratory Data Analysis ---\n")

# ── 3.1 Overall no-show rate ─────────────────────────────────────────────────
overall_rate <- mean(df$no_show_binary)
cat(sprintf("Overall no-show rate: %.1f%%\n", overall_rate * 100))

# ── 3.2 No-show rate by key variables ────────────────────────────────────────
cat("\nNo-show rate by SMS received:\n")
print(df %>% group_by(sms_received) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()))

cat("\nNo-show rate by age group:\n")
print(df %>% group_by(age_group) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()) %>% arrange(age_group))

cat("\nNo-show rate by day of week:\n")
print(df %>% group_by(appt_weekday) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()) %>% arrange(desc(no_show_rate)))

cat("\nNo-show rate by scholarship status:\n")
print(df %>% group_by(scholarship) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()))

cat("\nNo-show rate by alcoholism:\n")
print(df %>% group_by(alcoholism) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()))

cat("\nNo-show rate by high_risk flag:\n")
print(df %>% group_by(high_risk) %>%
        summarise(no_show_rate = round(mean(no_show_binary)*100,1),
                  n = n()))

# ── 3.3 PLOTS ─────────────────────────────────────────────────────────────────

# Plot 1: No-show rate by day of week
p1 <- df %>%
  group_by(appt_weekday) %>%
  summarise(rate = mean(no_show_binary) * 100) %>%
  mutate(appt_weekday = factor(appt_weekday,
                               levels = c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"))) %>%
  filter(!is.na(appt_weekday)) %>%
  ggplot(aes(x = appt_weekday, y = rate, fill = rate)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#a8d5e2", high = "#1F3864") +
  geom_text(aes(label = paste0(round(rate,1),"%")), vjust = -0.4, size = 3.5) +
  labs(title = "No-Show Rate by Day of Week",
       subtitle = "Mondays show the highest no-show rate",
       x = NULL, y = "No-Show Rate (%)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = "#1F3864"))

# Plot 2: No-show rate by age group
p2 <- df %>%
  group_by(age_group) %>%
  summarise(rate = mean(no_show_binary) * 100) %>%
  filter(!is.na(age_group)) %>%
  ggplot(aes(x = age_group, y = rate, fill = rate)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#a8d5e2", high = "#1F3864") +
  geom_text(aes(label = paste0(round(rate,1),"%")), vjust = -0.4, size = 3.5) +
  labs(title = "No-Show Rate by Age Group",
       subtitle = "Younger patients show higher no-show rates",
       x = NULL, y = "No-Show Rate (%)") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        plot.title = element_text(face = "bold", color = "#1F3864"))

# Plot 3: Lead time distribution by no-show
p3 <- df %>%
  filter(lead_time <= 90) %>%   # clip extreme outliers for visibility
  ggplot(aes(x = lead_time, fill = factor(no_show_binary))) +
  geom_histogram(bins = 40, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("#a8d5e2","#1F3864"),
                    labels = c("Showed Up","No-Show"),
                    name = NULL) +
  labs(title = "Lead Time Distribution by Outcome",
       subtitle = "Longer wait = higher no-show likelihood",
       x = "Days Between Scheduling & Appointment",
       y = "Count") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", color = "#1F3864"))

# Plot 4: SMS reminders by age group (interaction effect)
p4 <- df %>%
  filter(!is.na(age_group)) %>%
  group_by(age_group, sms_received) %>%
  summarise(rate = mean(no_show_binary) * 100, .groups = "drop") %>%
  mutate(sms_label = ifelse(sms_received == 1, "SMS Sent", "No SMS")) %>%
  ggplot(aes(x = age_group, y = rate, fill = sms_label)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("SMS Sent" = "#1F3864", "No SMS" = "#a8d5e2"),
                    name = NULL) +
  geom_text(aes(label = paste0(round(rate,1),"%")),
            position = position_dodge(width = 0.9), vjust = -0.3, size = 3) +
  labs(title = "SMS Effectiveness by Age Group",
       subtitle = "SMS reminders less effective for seniors",
       x = NULL, y = "No-Show Rate (%)") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        plot.title = element_text(face = "bold", color = "#1F3864"))

# Save all EDA plots
png("eda_plots.png", width = 1400, height = 1000, res = 120)
grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()
cat("EDA plots saved: eda_plots.png\n")


# =============================================================================
# SECTION 4 — CORRELATION ANALYSIS
# =============================================================================

cat("\n--- SECTION 4: Correlation Analysis ---\n")

numeric_features <- c("age","lead_time","scholarship","hypertension",
                      "diabetes","alcoholism","handicap","sms_received",
                      "prior_noshow_count","high_risk","no_show_binary")

corr_matrix <- cor(df[, numeric_features], use = "complete.obs")

cat("Correlations with no_show_binary (sorted):\n")
corr_with_target <- sort(corr_matrix[,"no_show_binary"], decreasing = TRUE)
print(round(corr_with_target, 3))

png("correlation_matrix.png", width = 1000, height = 900, res = 120)
corrplot(corr_matrix,
         method = "color",
         type   = "upper",
         tl.col = "#1F3864",
         tl.cex = 0.85,
         addCoef.col = "black",
         number.cex  = 0.65,
         col  = colorRampPalette(c("#a8d5e2","white","#1F3864"))(200),
         title = "Correlation Matrix — No-Show Predictors",
         mar  = c(0,0,2,0))
dev.off()
cat("Correlation matrix saved: correlation_matrix.png\n")

# =============================================================================
# SECTION 5 — MODEL PREPARATION
# =============================================================================

cat("\n--- SECTION 5: Model Preparation ---\n")

# ── 5.1 Select features for modeling ─────────────────────────────────────────
# Based on EDA and domain knowledge
model_features <- c(
  "age",
  "lead_time",
  "scholarship",
  "hypertension",
  "diabetes",
  "alcoholism",
  "handicap",
  "sms_received",
  "prior_noshow_count",
  "high_risk",
  "appt_weekday_num",
  "no_show_binary"
)

df_model <- df[, model_features] %>%
  filter(complete.cases(.))

cat("Modeling dataset:", nrow(df_model), "rows\n")

# ── 5.2 Train / Validation / Test split (60 / 20 / 20) ───────────────────────
n <- nrow(df_model)
train_idx <- sample(1:n, size = floor(0.6 * n))
remaining  <- setdiff(1:n, train_idx)
val_idx    <- sample(remaining, size = floor(0.5 * length(remaining)))
test_idx   <- setdiff(remaining, val_idx)

df_train <- df_model[train_idx, ]
df_val   <- df_model[val_idx,   ]
df_test  <- df_model[test_idx,  ]

cat(sprintf("Train: %d | Validation: %d | Test: %d\n",
            nrow(df_train), nrow(df_val), nrow(df_test)))

# ── 5.3 Target variable as factor for caret ───────────────────────────────────
df_train$no_show_binary <- factor(df_train$no_show_binary, labels = c("Showed","NoShow"))
df_val$no_show_binary   <- factor(df_val$no_show_binary,   labels = c("Showed","NoShow"))
df_test$no_show_binary  <- factor(df_test$no_show_binary,  labels = c("Showed","NoShow"))

# Repeat offender rate per patient
df <- df %>%
  group_by(patient_id) %>%
  mutate(
    total_prior_appts   = cumsum(lag(rep(1, n()), default = 0)),
    noshow_rate_patient = ifelse(total_prior_appts > 0,
                                 prior_noshow_count / total_prior_appts, 0)
  ) %>%
  ungroup()

# Is the appointment in the first hour of the day?
df$hour_scheduled <- as.integer(format(as.POSIXct(df$scheduled_day), "%H"))
df$appt_hour      <- as.integer(format(as.POSIXct(df$appointment_day), "%H"))

# Is it a same-day booking?
df$same_day <- as.integer(df$lead_time == 0)

# Is it a weekend appointment?
df$is_weekend <- as.integer(df$appt_weekday_num >= 6)

# Neighbourhood no-show rate (area-level risk)
neighbourhood_rates <- df %>%
  group_by(neighbourhood) %>%
  summarise(neighbourhood_noshow_rate = mean(no_show_binary))
df <- df %>% left_join(neighbourhood_rates, by = "neighbourhood")

# =============================================================================
# SECTION 6 — LOGISTIC REGRESSION MODEL
# =============================================================================

cat("\n--- SECTION 6: Logistic Regression ---\n")

# ── 6.1 Train full logistic regression model ──────────────────────────────────
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(42)
lr_model <- train(
  no_show_binary ~ .,
  data      = df_train,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)

cat("\nLogistic Regression — Cross-Validation Results:\n")
print(lr_model$results)

# ── 6.2 Odds Ratios ───────────────────────────────────────────────────────────
cat("\nOdds Ratios (exponentiated coefficients):\n")
lr_coefs    <- coef(lr_model$finalModel)
lr_or       <- exp(lr_coefs)
lr_ci       <- exp(confint(lr_model$finalModel))
lr_or_table <- data.frame(
  Variable  = names(lr_coefs),
  OddsRatio = round(lr_or, 3),
  CI_Lower  = round(lr_ci[,1], 3),
  CI_Upper  = round(lr_ci[,2], 3)
) %>% arrange(desc(OddsRatio))
print(lr_or_table)

# ── 6.3 Validation performance ────────────────────────────────────────────────
lr_val_probs <- predict(lr_model, newdata = df_val, type = "prob")[,"NoShow"]
lr_val_class <- predict(lr_model, newdata = df_val)

cat("\nValidation Confusion Matrix:\n")
lr_cm_val <- confusionMatrix(lr_val_class, df_val$no_show_binary, positive = "NoShow")
print(lr_cm_val)

lr_val_auc <- roc(
  response  = df_val$no_show_binary,
  predictor = lr_val_probs,
  levels    = c("Showed","NoShow")
)
cat(sprintf("Validation AUC: %.4f\n", auc(lr_val_auc)))

# ── 6.4 Test performance ──────────────────────────────────────────────────────
lr_test_probs <- predict(lr_model, newdata = df_test, type = "prob")[,"NoShow"]
lr_test_class <- predict(lr_model, newdata = df_test)

cat("\nTest Confusion Matrix:\n")
lr_cm_test <- confusionMatrix(lr_test_class, df_test$no_show_binary, positive = "NoShow")
print(lr_cm_test)

lr_test_auc <- roc(
  response  = df_test$no_show_binary,
  predictor = lr_test_probs,
  levels    = c("Showed","NoShow")
)
cat(sprintf("Test AUC: %.4f\n", auc(lr_test_auc)))

# Install if needed: install.packages("randomForest")
library(randomForest)

set.seed(42)
rf_model <- train(
  no_show_binary ~ .,
  data      = df_train,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = data.frame(mtry = c(3, 5, 7))
)

rf_val_probs <- predict(rf_model, newdata = df_val, type = "prob")[,"NoShow"]
rf_val_auc   <- roc(
  response  = df_val$no_show_binary,
  predictor = rf_val_probs,
  levels    = c("Showed","NoShow")
)
cat(sprintf("Random Forest Validation AUC: %.4f\n", auc(rf_val_auc)))


# Install if needed: 
library(xgboost)

# Prepare matrices
train_matrix <- xgb.DMatrix(
  data  = as.matrix(df_train %>% select(-no_show_binary)),
  label = as.integer(df_train$no_show_binary == "NoShow")
)

val_matrix <- xgb.DMatrix(
  data  = as.matrix(df_val %>% select(-no_show_binary)),
  label = as.integer(df_val$no_show_binary == "NoShow")
)

# Train
set.seed(42)
xgb_direct <- xgb.train(
  params  = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = 6,
    eta              = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  ),
  data    = train_matrix,
  nrounds = 100,
  verbose = 1
)

# Evaluate
xgb_val_probs <- predict(xgb_direct, val_matrix)
xgb_val_auc   <- roc(
  response  = as.integer(df_val$no_show_binary == "NoShow"),
  predictor = xgb_val_probs
)
cat(sprintf("XGBoost Validation AUC: %.4f\n", auc(xgb_val_auc)))


# Install if needed: 
install.packages("themis")
library(themis)

ctrl_smote <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "smote"   # add this line
)

# Then retrain your logistic regression or RF with ctrl_smote instead of ctrl

set.seed(42)
rf_smote <- train(
  no_show_binary ~ .,
  data      = df_train,
  method    = "rf",
  trControl = ctrl_smote,
  metric    = "ROC",
  tuneGrid  = data.frame(mtry = c(3, 5))
)

rf_smote_probs <- predict(rf_smote, newdata = df_val, type = "prob")[,"NoShow"]
rf_smote_auc   <- roc(
  response  = df_val$no_show_binary,
  predictor = rf_smote_probs,
  levels    = c("Showed","NoShow")
)
cat(sprintf("Random Forest + SMOTE Validation AUC: %.4f\n", auc(rf_smote_auc)))

set.seed(42)
lr_smote <- train(
  no_show_binary ~ .,
  data      = df_train,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl_smote,
  metric    = "ROC"
)

lr_smote_probs <- predict(lr_smote, newdata = df_val, type = "prob")[,"NoShow"]
lr_smote_auc   <- roc(
  response  = df_val$no_show_binary,
  predictor = lr_smote_probs,
  levels    = c("Showed","NoShow")
)
cat(sprintf("Logistic Regression + SMOTE Validation AUC: %.4f\n", auc(lr_smote_auc)))

# =============================================================================
# SECTION 7 — STEPWISE LOGISTIC REGRESSION
# =============================================================================

cat("\n--- SECTION 7: Stepwise Logistic Regression ---\n")

# Fit base glm for stepwise
lr_base <- glm(
  no_show_binary ~ .,
  data   = df_train %>% mutate(no_show_binary = as.integer(no_show_binary == "NoShow")),
  family = binomial
)

lr_step <- step(lr_base, direction = "both", trace = 0)

cat("\nStepwise selected variables:\n")
print(names(coef(lr_step)))

# Stepwise validation AUC
stepwise_train_data <- df_train %>%
  mutate(no_show_binary = as.integer(no_show_binary == "NoShow"))
stepwise_val_data <- df_val %>%
  mutate(no_show_binary = as.integer(no_show_binary == "NoShow"))

lr_step_probs <- predict(lr_step, newdata = stepwise_val_data, type = "response")
lr_step_auc   <- roc(stepwise_val_data$no_show_binary, lr_step_probs)
cat(sprintf("Stepwise Logistic Regression Validation AUC: %.4f\n", auc(lr_step_auc)))


# =============================================================================
# SECTION 8 — ROC CURVES & COMPARISON PLOT
# =============================================================================

cat("\n--- SECTION 8: ROC Curve Plots ---\n")

# ROC: Full vs Stepwise logistic regression (validation)
png("roc_logistic_comparison.png", width = 900, height = 750, res = 120)
plot(lr_val_auc,
     col  = "#1F3864",
     lwd  = 2.5,
     main = "ROC Curve — Full vs Stepwise Logistic Regression\n(Validation Set)",
     cex.main = 1.1)
plot(lr_step_auc, add = TRUE, col = "#e07b39", lwd = 2.5, lty = 2)
legend("bottomright",
       legend = c(
         sprintf("Full LR (AUC = %.3f)", auc(lr_val_auc)),
         sprintf("Stepwise LR (AUC = %.3f)", auc(lr_step_auc))
       ),
       col    = c("#1F3864","#e07b39"),
       lwd    = 2.5,
       lty    = c(1,2),
       bty    = "n",
       cex    = 1.05)
abline(a = 0, b = 1, col = "gray70", lty = 3)
dev.off()
cat("ROC comparison plot saved: roc_logistic_comparison.png\n")

# ROC: Train vs Validation (overfitting check)
lr_train_probs <- predict(lr_model, newdata = df_train, type = "prob")[,"NoShow"]
lr_train_auc   <- roc(
  response  = df_train$no_show_binary,
  predictor = lr_train_probs,
  levels    = c("Showed","NoShow")
)

png("roc_train_vs_validation.png", width = 900, height = 750, res = 120)
plot(lr_train_auc,
     col  = "#1F3864",
     lwd  = 2.5,
     main = "ROC Curve — Training vs Validation\n(Overfitting Check)",
     cex.main = 1.1)
plot(lr_val_auc, add = TRUE, col = "#e07b39", lwd = 2.5, lty = 2)
legend("bottomright",
       legend = c(
         sprintf("Training (AUC = %.3f)", auc(lr_train_auc)),
         sprintf("Validation (AUC = %.3f)", auc(lr_val_auc))
       ),
       col  = c("#1F3864","#e07b39"),
       lwd  = 2.5,
       lty  = c(1,2),
       bty  = "n",
       cex  = 1.05)
abline(a = 0, b = 1, col = "gray70", lty = 3)
dev.off()
cat("Train vs Validation ROC saved: roc_train_vs_validation.png\n")


# =============================================================================
# SECTION 9 — THRESHOLD TUNING & RISK STRATIFICATION
# =============================================================================

cat("\n--- SECTION 9: Threshold Tuning & Risk Stratification ---\n")

# ── 9.1 Find optimal threshold (maximizes sensitivity + specificity) ──────────
thresholds <- seq(0.1, 0.9, by = 0.01)
val_actual <- as.integer(df_val$no_show_binary == "NoShow")

threshold_results <- data.frame(
  threshold   = thresholds,
  sensitivity = NA,
  specificity = NA,
  f1          = NA
)

for (i in seq_along(thresholds)) {
  preds  <- as.integer(lr_val_probs >= thresholds[i])
  tp     <- sum(preds == 1 & val_actual == 1)
  fp     <- sum(preds == 1 & val_actual == 0)
  tn     <- sum(preds == 0 & val_actual == 0)
  fn     <- sum(preds == 0 & val_actual == 1)
  sens   <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  spec   <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  prec   <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  f1     <- ifelse((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0)
  threshold_results[i, "sensitivity"] <- sens
  threshold_results[i, "specificity"] <- spec
  threshold_results[i, "f1"]          <- f1
}

# Best threshold by F1
best_thresh <- threshold_results$threshold[which.max(threshold_results$f1)]
cat(sprintf("Optimal threshold (max F1): %.2f\n", best_thresh))
cat("Threshold performance at optimum:\n")
print(threshold_results[threshold_results$threshold == best_thresh, ])

# ── 9.2 Plot sensitivity/specificity tradeoff ────────────────────────────────
png("threshold_tuning.png", width = 950, height = 650, res = 120)
plot(threshold_results$threshold, threshold_results$sensitivity,
     type = "l", col = "#1F3864", lwd = 2.5,
     ylim = c(0, 1),
     xlab = "Probability Threshold",
     ylab = "Metric Value",
     main = "Sensitivity vs Specificity vs F1 — Threshold Tuning")
lines(threshold_results$threshold, threshold_results$specificity,
      col = "#e07b39", lwd = 2.5, lty = 2)
lines(threshold_results$threshold, threshold_results$f1,
      col = "#2e8b57", lwd = 2.5, lty = 3)
abline(v = best_thresh, col = "gray50", lty = 4, lwd = 1.5)
legend("topright",
       legend = c("Sensitivity","Specificity","F1",
                  paste("Optimal threshold =", best_thresh)),
       col    = c("#1F3864","#e07b39","#2e8b57","gray50"),
       lwd    = c(2.5, 2.5, 2.5, 1.5),
       lty    = c(1, 2, 3, 4),
       bty    = "n", cex = 0.95)
dev.off()
cat("Threshold tuning plot saved: threshold_tuning.png\n")

# ── 9.3 Risk stratification buckets ──────────────────────────────────────────
df_val_risk <- df_val %>%
  mutate(
    noshow_prob = lr_val_probs,
    risk_tier   = case_when(
      noshow_prob >= 0.50 ~ "High Risk (≥50%)",
      noshow_prob >= 0.30 ~ "Medium Risk (30–50%)",
      TRUE                ~ "Low Risk (<30%)"
    ),
    actual_noshow = as.integer(no_show_binary == "NoShow")
  )

cat("\nRisk Tier Summary:\n")
risk_summary <- df_val_risk %>%
  group_by(risk_tier) %>%
  summarise(
    n              = n(),
    actual_noshows = sum(actual_noshow),
    noshow_rate    = round(mean(actual_noshow) * 100, 1),
    avg_prob       = round(mean(noshow_prob) * 100, 1)
  ) %>% arrange(desc(noshow_rate))
print(risk_summary)

# ── 9.4 Business impact calculation ──────────────────────────────────────────
cat("\n--- Business Impact Estimation ---\n")
cost_per_noshow     <- 200    # $ per missed appointment (industry benchmark)
annual_appointments <- 1e9   # global annual appointments (conservative proxy)
baseline_noshow_rate <- overall_rate

high_risk_patients    <- risk_summary %>% filter(grepl("High", risk_tier))
intervention_reduction <- 0.20   # assume 20% reduction with targeted outreach

cat(sprintf("Baseline no-show rate: %.1f%%\n", baseline_noshow_rate * 100))
cat(sprintf("Cost per no-show: $%d\n", cost_per_noshow))
cat(sprintf("Annual appointments (proxy): {:,.0f}\n"))   # informational

# Per-clinic estimate (medium clinic: ~30,000 appointments/year)
clinic_appts        <- 30000
clinic_noshows      <- clinic_appts * baseline_noshow_rate
clinic_cost         <- clinic_noshows * cost_per_noshow
clinic_savings      <- clinic_cost * intervention_reduction

cat(sprintf("\nFor a medium-sized clinic (%d appts/year):\n", clinic_appts))
cat(sprintf("  Estimated annual no-shows:  %d\n", round(clinic_noshows)))
cat(sprintf("  Estimated annual cost:      $%s\n",
            format(round(clinic_cost), big.mark=",")))
cat(sprintf("  Projected savings (20%% reduction): $%s\n",
            format(round(clinic_savings), big.mark=",")))


# =============================================================================
# SECTION 10 — ODDS RATIO VISUALIZATION
# =============================================================================

cat("\n--- SECTION 10: Odds Ratio Forest Plot ---\n")

# Clean OR table for plotting (remove intercept)
or_plot_data <- lr_or_table %>%
  filter(Variable != "(Intercept)") %>%
  mutate(
    Variable = recode(Variable,
                      "age"               = "Age",
                      "lead_time"         = "Lead Time (days)",
                      "scholarship"       = "Scholarship",
                      "hypertension"      = "Hypertension",
                      "diabetes"          = "Diabetes",
                      "alcoholism"        = "Alcoholism",
                      "handicap"          = "Handicap",
                      "sms_received"      = "SMS Received",
                      "prior_noshow_count"= "Prior No-Shows",
                      "high_risk"         = "High-Risk Flag",
                      "appt_weekday_num"  = "Day of Week"
    ),
    direction = ifelse(OddsRatio > 1, "Increases Risk", "Decreases Risk")
  )

png("odds_ratios.png", width = 950, height = 700, res = 120)
ggplot(or_plot_data, aes(x = OddsRatio, y = reorder(Variable, OddsRatio),
                         color = direction)) +
  geom_point(size = 3.5) +
  geom_errorbarh(aes(xmin = CI_Lower, xmax = CI_Upper), height = 0.25, lwd = 1) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("Increases Risk" = "#1F3864",
                                "Decreases Risk" = "#e07b39"),
                     name = NULL) +
  labs(title = "Odds Ratios — Logistic Regression Coefficients",
       subtitle = "OR > 1: increases no-show risk  |  OR < 1: decreases no-show risk",
       x = "Odds Ratio (with 95% CI)", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", color = "#1F3864"),
    plot.subtitle = element_text(color = "gray50"),
    legend.position = "bottom"
  )
dev.off()
cat("Odds ratio plot saved: odds_ratios.png\n")


# =============================================================================
# SECTION 11 — FINAL MODEL SUMMARY
# =============================================================================

cat("\n=== FINAL MODEL SUMMARY ===\n")
cat(sprintf("Model:               Logistic Regression (5-fold CV)\n"))
cat(sprintf("Dataset:             110,527 appointments | 22.8%% no-show rate\n"))
cat(sprintf("Train/Val/Test:      60%% / 20%% / 20%%\n"))
cat(sprintf("Validation AUC:      %.4f\n", auc(lr_val_auc)))
cat(sprintf("Test AUC:            %.4f\n", auc(lr_test_auc)))
cat(sprintf("Optimal Threshold:   %.2f\n", best_thresh))
cat(sprintf("Top Risk Drivers:    Alcoholism, Scholarship, Prior No-Shows, Lead Time\n"))
cat(sprintf("Protective Factors:  SMS Received (for younger patients), Hypertension, Age\n"))
cat(sprintf("Projected Savings:   $%s/year per medium clinic\n",
            format(round(clinic_savings), big.mark=",")))

cat("\n=== FILES GENERATED ===\n")
cat("  eda_plots.png\n")
cat("  correlation_matrix.png\n")
cat("  roc_logistic_comparison.png\n")
cat("  roc_train_vs_validation.png\n")
cat("  threshold_tuning.png\n")
cat("  odds_ratios.png\n")

cat("\nAnalysis complete.\n")
