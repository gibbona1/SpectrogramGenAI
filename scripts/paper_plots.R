library(xtable)
library(dplyr)
#take in dataframe, calc train test val counts, distributions, make latex table with colourbars
library(reticulate)
library(ggplot2)
np <- import("numpy")
setwd('../../Downloads/')
data_npz <- np$load("spectrogram_data/specdata.npz", allow_pickle = TRUE)

df_cols <- colnames(read.csv('data/model_output_loc_merge.csv'))
train_df <- as.data.frame(data_npz$get("train_df"))
colnames(train_df) <- c("index0", df_cols)
val_df <- as.data.frame(data_npz$get("test_df"))
colnames(val_df) <- c("index0", df_cols)

pop_classes <- data_npz$get("categories")

train_df$common_name <- unlist(train_df$common_name)
val_df$common_name <- unlist(val_df$common_name)

train_c <- train_df %>%
  group_by(common_name) %>%
  summarise(train_counts = n())

val_c <- val_df %>%
  group_by(common_name) %>%
  summarise(val_counts = n())

rbind(train_df, val_df) |>
  mutate(train_test_split = c(rep("train", nrow(train_df)), rep("val", nrow(val_df)))) |>
  group_by(common_name, train_test_split) |>
  summarise(count = n()) |>
  mutate(proportion = count/sum(count)) |>
  ggplot(aes(x=common_name, y=proportion, fill=train_test_split)) + geom_histogram(stat="identity", position = "dodge")

get_neal_data <- function(lab_file, wav_path, classes) {
  # Read the CSV file
  df <- read.csv(lab_file)
  
  # Copy 'class_label' to 'common_name'
  df$common_name <- df$class_label
  
  # Filter based on file names present in the wav_path directory
  wav_files <- list.files(wav_path)
  df_sub <- df %>%
    filter(file_name %in% wav_files,
    # Further filter based on 'common_name' being in the provided 'classes'
    common_name %in% classes,
    labeler %in% c("dk", "hh", "iw", "ms"),   # Only specific labelers
    confidence >= 0.9,
    ) %>%
    mutate(start_time = as.integer(start_time)) %>%
    distinct(file_name, start_time, .keep_all = TRUE)
  return(df_sub)
}

test_df <- get_neal_data('data/neal_labels_remapped.csv', '../Documents/GitHub/neal_data/data', pop_classes)

test_c <- test_df %>%
  group_by(common_name) %>%
  summarise(test_counts = n())

df <- merge(merge(train_c, val_c, by = 'common_name'), test_c, by = 'common_name', all.x = TRUE)

df[is.na(df)] <- 0L
#df <- data.frame(species      = paste("Species", LETTERS[1:4]),
#                 train_counts = round(200*c(0.4, 0.3, 0.1, 0.2)),
#                 val_counts   = round(50*c(0.3, 0.35, 0.05, 0.3)),
#                 test_counts  = round(50*c(0.2, 0.2, 0.5, 0.1))
#)

df$train <- paste0("\\color{seabornBlue}{\\rule{", round(6*df$train_counts/sum(df$train_counts), 3), "cm}{6pt} ",
                   round(100*df$train_counts/sum(df$train_counts), 1), "}")
df$val  <- paste0("\\color{seabornOrange}{\\rule{", round(6*df$val_counts/sum(df$val_counts), 3), "cm}{6pt} ",
                  round(100*df$val_counts/sum(df$val_counts), 1), "}")
df$test <- paste0("\\color{seabornGreen}{\\rule{", round(2.5*df$test_counts/sum(df$test_counts), 3), "cm}{6pt} ",
                  round(100*df$test_counts/sum(df$test_counts), 1), "}")

colnames(df) <- gsub("_", " ", colnames(df))

print(xtable(df), sanitize.text.function = identity)

