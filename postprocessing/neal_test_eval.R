# Load necessary libraries
library(dplyr)
library(stringr)
library(readr)
library(purrr)
library(reticulate)
library(ggplot2)

# Define the folder containing the files
fpath <- "C:\\Users\\Anthony\\Downloads\\data\\neal_data_emb"

# Get list of files
files <- list.files(fpath, pattern = "\\.embeddings.txt$", full.names = TRUE)

# Function to process each file
process_file <- function(fpath) {
  # Extract the filename
  file_name <- basename(fpath)

  # Remove the '_ext_{number}' part from filename and get the start time
  fname_clean <- str_remove(file_name, "_ext_\\d+\\.birdnet\\.embeddings.txt$")
  fname_clean <- paste0(fname_clean, ".wav")
  fstart_time <- str_extract(file_name, "(?<=_ext_)\\d+")

  # Read the contents of the file (tab-separated)
  file_content <- read_tsv(fpath, col_names = FALSE, col_types = cols())[, 3]

  colnames(file_content) <- c("embeddings")

  # Convert embeddings column (comma-separated values) to a character string
  file_content <- file_content %>%
    mutate(
      file_name = fname_clean,
      start_time = as.integer(fstart_time),
      embeddings = as.character(.data$embeddings)
    )

  file_content
}

# Apply the function to all files and combine into a single data frame
result_df <- map_dfr(files, process_file)

# Display the resulting data frame
result_df


# Load your metadata (assuming it's in a CSV or similar format)
metadata <- read.csv("datasets\\neal_labels_remapped.csv")

# Define the folder paths
neal_folder <- "C:\\Users\\Anthony\\Documents\\GitHub\\neal_data\\data"

np <- import("numpy")
specdata <-  np$load("datasets\\specdata.npz", allow_pickle = TRUE)

classes <- specdata["categories"]

# Step 1: Filter data based on criteria
filtered_metadata <- metadata %>%
  filter(file_name %in% list.files(neal_folder),   # Only files in neal_folder
         labeler %in% c("dk", "hh", "iw", "ms"),   # Only specific labelers
         confidence >= 0.9,
         class_label %in% classes) %>%
  mutate(start_time = as.integer(start_time)) %>%
  distinct(file_name, start_time, .keep_all = TRUE)

birdnet_df <- result_df %>%
  arrange(file_name, start_time) %>%
  select(file_name, start_time, embeddings)

neal_df <- filtered_metadata %>%
  arrange(file_name, start_time) %>%
  select(file_name, start_time, class_label)

label_path <- "..\\BirdNET-Analyzer\\checkpoints\\V2.3\\BirdNET_GLOBAL_3K_V2.3_Labels.txt"

# Step 1: Load the label file and extract class names
class_names <- readLines(label_path) %>%
  sapply(function(x) str_split(x, "_")[[1]][2])

# Step 2: Get indices of matching class names
matching_indices <- match(classes, class_names)

# Step 3: Find the max index of matching classes for each row in your data frame
# Assuming your main data frame is called `result_df`
birdnet_df <- birdnet_df %>%
  rowwise() %>%
  mutate(
    pred_idx = str_split(embeddings, ",")[[1]] %>% as.numeric(),
    pred_class = class_names[which.max(pred_idx)],
    pred_class_mask = classes[which.max(pred_idx %>% .[matching_indices])]
  ) %>%
  ungroup()


#birdnet_df$
# Display the resulting data frame
neal_df

merge_df <- birdnet_df %>%
  select(file_name, start_time, pred_class, pred_class_mask) %>%
  merge(neal_df, by = c("file_name", "start_time"))

(merge_df$pred_class == merge_df$class_label) %>% mean()

(merge_df$pred_class_mask == merge_df$class_label) %>% mean()

confusion_matrix <- merge_df %>%
  count(class_label, pred_class_mask) %>%
  as.data.frame()

ggplot(data = confusion_matrix,
       mapping = aes(x = Var1,
                     y = Var2)) +
  geom_tile(aes(fill = Freq)) +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 0) +
  scale_fill_gradient(low = "blue",
                      high = "red",
                      trans = "log")

bar_df <- data.frame(
  model = c("BirdNET", "BirdNET_mask",
            "Ensemble_0", "Ensemble_250",
            "L_Ensemble_0", "L_Ensemble_250"),
  accuracy = c(0.5640051, 0.7718631, 0.5261, 0.5588, 0.6000, 0.6400)
)

bar_df$acc <- bar_df$accuracy
bar_df$superset <- factor(c("BirdNET", "BirdNET",
                            "Ensemble", "Ensemble",
                            "Large Ensemble", "Large Ensemble"),
                          levels = c("BirdNET", "Ensemble", "Large Ensemble"))
bar_df$subset <- c("base", "mask",
                   "0 synthetic", "250 synthetic",
                   "0 synthetic", "250 synthetic")

library(ggplot2)

p <- ggplot(bar_df, aes(x = subset, y = 100 * acc)) +
  geom_bar(fill = c("darkred"), stat = "identity") +
  geom_text(aes(label = sprintf("%0.1f", round(100 * acc, 1)),
                y = 100 * acc, vjust = -0.5),
            size = 6, color = "black") +
  facet_grid(. ~ superset, scales = "free_x", space = "free_x") +
  #scale_x_discrete(expand = c(0.25, 0.15)) +
  scale_y_continuous(expand = c(0, 0.05)) +
  coord_cartesian(ylim = c(0, 90)) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = "black"),
    panel.border = element_rect(fill = NA, color = "black"),
    panel.spacing = unit(0.5, "lines"),
    axis.text = element_text(size = 20),
    axis.title = element_text(size = 22),
    axis.title.x = element_blank(),
    axis.ticks = element_line(),
    strip.text.x = element_text(size = 28),
    title = element_text(size = 20)
  ) +
  labs(y = "Accuracy (%)")
p
ggsave("test_eval.png", plot = p, width = 18, height = 6, dpi = 300, units = "in", bg = "white")
