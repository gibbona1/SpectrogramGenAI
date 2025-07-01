library(dplyr)
library(ggplot2)
library(stringr)
library(knitr)
library(kableExtra)
library(tidyr)

folder <- "results"
res_files <- list.files(folder)
knowledge_dist <- FALSE
res_files <- res_files[grep("([^_]+)_synth(0|[1-2]?50)*.csv$", res_files)]

res_ncol <- sapply(res_files, function(x) ncol(read.csv(file.path(folder, x))))

read_csv_regex <- function(x) {
  aindex <- str_extract(x, "noind|aind")
  aindex_samp <- str_sub(str_extract(x, "_(in|mvn)"), start = 2)

  df <- read.csv(file.path(folder, x))

  df$aindex <- aindex
  df$aindex_samp <- aindex_samp
  df
}

res_df <- do.call(rbind, purrr::map(names(res_ncol), read_csv_regex))

res_df_max <- res_df %>%
  group_by(Name, Synthetic) %>%
  slice(which.max(Val.Accuracy))


res_df_max %>%
  arrange(Name == "ensemble") %>%
  mutate(
    Model = factor(Name, levels = c(unique(res_df_max$Name) %>% .[-which(. == "ensemble")], "ensemble")),
    `Val.Top.1.Accuracy` = 1 - `Val.Top.1.Error`,
    `Val.Top.5.Accuracy` = 1 - `Val.Top.5.Error`
  ) %>%
  pivot_longer(cols = c(Val.Top.1.Accuracy, Val.Top.5.Accuracy),
               names_to = "Metric",
               values_to = "Value") %>%
  ggplot(aes(x = factor(Synthetic), y = 100 * Value)) +
  geom_line(data = . %>% filter(Metric == "Val.Top.1.Accuracy"),
            aes(col = Metric, group = Model)) +
  geom_point(data = . %>% filter(Metric == "Val.Top.1.Accuracy"),
             aes(col = Metric, fill = Metric, group = Model)) +
  #scale_y_continuous(labels = scales::percent) +
  scale_y_continuous(breaks = seq(75, 100, 5),
                     minor_breaks = seq(75, 100, 1),
                     expand = c(0, 0)) +
  coord_cartesian(ylim = c(75, 100.5)) +
  scale_x_discrete(expand = c(0.075, 0.075)) +
  geom_line(data = . %>% filter(Metric == "Val.Top.5.Accuracy"),
            aes(col = Metric, group = Model)) +
  geom_point(data = . %>% filter(Metric == "Val.Top.5.Accuracy"),
             aes(col = Metric, fill = Metric, group = Model)) +
  facet_wrap(~ Model, ncol = 5) +
  scale_colour_manual(values = c("Val.Top.5.Accuracy" = "#00b8e7",
                                 "Val.Top.1.Accuracy" = "#f8766d"),
                      labels = c("Top-1", "Top-5")) +
  guides(alpha = "none", fill = "none", col = guide_legend(reverse = TRUE)) +
  #geom_text(aes(label = scales::percent(Value, accuracy = 0.1), y = Value),
  #          position = position_dodge(width = 1), vjust=-0.5, size=4) +
  geom_text(data = . %>%
              group_by(Model, Metric) %>%
              arrange(Value) %>%
              slice(c(1, n())) %>%
              mutate(vjst = ifelse(row_number() == 1, 1.2, -0.5)),
            aes(label = round(100 * Value, 1), y = 100 * Value, vjust = vjst),
            size = 6.5, color = "black") +
  theme_minimal() +
  theme(panel.border = element_rect(fill = 0),
        panel.spacing = unit(0.5, "lines"),
        axis.text = element_text(size = 22),
        axis.title = element_text(size = 26),
        axis.ticks = element_line(size = 1),
        strip.text.x = element_text(size = 28),
        legend.title = element_text(size = 22),
        legend.key.size = unit(1.25, "cm"),
        legend.text = element_text(size = 18),
        legend.box.background = element_rect(colour = "black", size = 1),
        legend.position = c(.96, .15)) +
  labs(x = "Number of synthetic examples included per class", y = "Accuracy (%)", fill = "Model")


res_df_max %>%
  mutate(
    Model = Name,  # Rename "Name" to "Model"
    `Test.Top.1.Accuracy` = 1 - `Test.Top.1.Error`,
    `Test.Top.5.Accuracy` = 1 - `Test.Top.5.Error`
  ) %>%
  pivot_longer(cols = c(Test.Top.1.Accuracy, Test.Top.5.Accuracy),
               names_to = "Metric",
               values_to = "Value") %>%
  ggplot(aes(x = factor(Synthetic), y = 100 * Value)) +
  geom_line(data = . %>% filter(Metric == "Test.Top.1.Accuracy"),
            aes(col = Model, alpha = Metric, group = Model)) +
  geom_point(data = . %>% filter(Metric == "Test.Top.1.Accuracy"),
             aes(col = Model, alpha = Metric, group = Model)) +
  scale_x_discrete(expand = c(0.075, 0.075)) +
  geom_line(data = . %>% filter(Metric == "Test.Top.5.Accuracy"),
            aes(col = Model, alpha = Metric, group = Model)) +
  geom_point(data = . %>% filter(Metric == "Test.Top.5.Accuracy"),
             aes(col = Model, alpha = Metric, group = Model)) +
  facet_wrap(~ Model, ncol = 5) +
  scale_alpha_manual(values = c("Test.Top.5.Accuracy" = 0.3,
                                "Test.Top.1.Accuracy" = 1.0),
                     labels = c("Top-1", "Top-5")) +
  guides(fill="none", col = "none", alpha = guide_legend(reverse = TRUE)) +
  geom_text(data = . %>%
              group_by(Model, Metric) %>%
              arrange(Value) %>%
              slice(c(1, n())) %>%
              mutate(vjst = ifelse(row_number() == 1, -0.5, 0.5)),
            aes(label = round(100 * Value, 1), y = 100 * Value, vjust = vjst),
            size = 5.5, color = "black") +
  theme_minimal() +
  theme(panel.border = element_rect(fill = 0),
        panel.spacing = unit(0.5, "lines"),
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 22),
        axis.ticks = element_line(size = 1),
        strip.text.x = element_text(size = 28),
        legend.title = element_text(size = 22),
        legend.key.size = unit(0.95, "cm"),
        legend.text = element_text(size = 18),
        legend.box.background = element_rect(colour = "black", size = 1),
        legend.position = c(.963, .12)) +
  labs(x = "Synthetic", y = "Accuracy (%)", fill = "Model")

res_df_max %>%
  filter(Synthetic %in% c(0, 250)) %>%
  select(Name, Synthetic, Val.Accuracy, Test.Accuracy)

res_df %>%
  mutate(
    `Top-1` = 1 - `Val.Top.1.Error`,
    `Top-5` = 1 - `Val.Top.5.Error`,
    Accuracy = Val.Accuracy,
    Precision = Val.Precision,
    Recall = Val.Recall,
    F1 = Val.F1
  ) %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1, `Top-5`),
               names_to = "Metric",
               values_to = "Value") %>%
  #sort by name, synthetic, epoch
  arrange(Name, Synthetic, Epoch) %>%
  ggplot(aes(x = Epoch, y = Value, colour = as.factor(Synthetic))) +
  geom_line() +
  theme_minimal() +
  theme(panel.border = element_rect(fill = 0)) +
  theme(axis.text = element_text(size = 12),
        axis.title = element_text(size = 18),
        strip.text.x = element_text(size = 15),
        strip.text.y = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.key.size = unit(0.75, "cm"),
        legend.text = element_text(size = 12)) +
  facet_grid(Metric ~ Name, scales = "free_y") +
  labs(colour = "Synthetic")

res_df_max %>%
  pivot_longer(cols = c(Val.Accuracy, Val.Precision, Val.Recall, Val.F1, Val.Top.1.Error, Val.Top.5.Error),
               names_to = "Metric",
               values_to = "Value") %>%
  ggplot(aes(x = Synthetic, y = Value, colour = Name)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(x = "Synthetic", y = "Value", colour = "Model") +
  theme_minimal() +
  theme(panel.border = element_rect(fill = 0)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


formatted_data <- res_df_max %>%
  ungroup() %>%
  mutate(
    Model = Name,  # Rename "Name" to "Model"
    `Synthetic (per class)` = as.character(Synthetic),
    Loss = formatC(Val.Loss, format = "f", digits = 3),  # Format loss to 3dp
    Accuracy = Val.Accuracy * 100,  # Convert to percentage (for comparison)
    Precision = Val.Precision * 100,  # Convert to percentage (for comparison)
    Recall = Val.Recall * 100,  # Convert to percentage (for comparison)
    F1 = Val.F1 * 100,  # Convert to percentage (for comparison)
    `Top-1 Error` = (1 - Val.Top.1.Error) * 100,
    `Top-5 Error` = (1 - Val.Top.5.Error) * 100
  ) %>%
  select(Model, `Synthetic (per class)`, Loss, Accuracy, Precision, Recall, F1, `Top-1 Error`, `Top-5 Error`)

# Step 2: Identify the maximum values in each group and bold them
# Step 2: Identify the maximum values in each group and bold them
bold_max <- function(x) {
  ifelse(x == max(x),
         paste0("\\textbf{", formatC(x, format = "f", digits = 2), "}"),
         formatC(x, format = "f", digits = 2))
}

formatted_data <- formatted_data %>%
  group_by(Model) %>%
  mutate(across(c(Accuracy, Precision, Recall, F1, `Top-1 Error`, `Top-5 Error`),
                bold_max)) %>%
  select(-c(`Top-1 Error`, Loss)) %>%
  select(Model, `Synthetic (per class)`, Precision, Recall, F1, Accuracy,
         `Top-5 Error`) %>%
  ungroup()

# Step 3: Convert to LaTeX table with kableExtra
latex_table <- kable(formatted_data, format = "latex",
                     booktabs = TRUE, escape = FALSE,
                     col.names = c("Model", "Synth", "Accuracy (\\%)",
                                   "Precision", "Recall", "F1", "Top-5")) %>%
  kable_styling(latex_options = c("striped", "hold_position"))

# Print the LaTeX table
cat(latex_table)
