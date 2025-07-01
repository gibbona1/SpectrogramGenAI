# Load libraries
library(ggplot2)
library(dplyr)
library(lubridate)

# Example dataset
data <- read.csv("datasets/model_output_loc_merge.csv")

ggplot(data, aes(x = confidence)) +
  geom_histogram(bins = 30, fill = "skyblue1", colour = "grey40") +
  labs(x = "Confidence", y = "Count") +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 10000)) +
  theme_bw()

# Step 1: Extract datetime, date and hour
data <- data %>%
  mutate(datetime = as.POSIXct(datetime, format = "%Y-%m-%dT%H:%M:%S")) %>%
  mutate(date = as.Date(datetime),
         hour = hour(floor_date(datetime, unit = "hour")))

# Step 2: Group by date and hour, then count occurrences
data_count <- data %>%
  mutate(recorder = case_when(
    recorder == "CARNSOREMET" ~ "Carnsore",
    recorder == "CLOOSHVALLEY" ~ "Cloosh",
    recorder == "RAHORA" ~ "Rahora",
    recorder == "RICHFIELDM1" ~ "Richfield",
    recorder == "TEEVURCHER" ~ "Teevurcher",
    TRUE ~ recorder  # In case there are other values you don't want to change
  )) %>%
  group_by(date, hour, recorder) %>%
  summarise(count = n()) %>%
  ungroup() %>%
  group_by(recorder) %>%
  mutate(
    min_date = floor_date(min(date), "month"),
    max_date = ceiling_date(max(date), "month") - days(1)
  ) %>%
  ungroup()

pb <- scales::pretty_breaks(n = 5)

# Step 3: Plot with ggplot
h_breaks <- seq(0, 24, by = 1)

data_count %>%
  filter(recorder == "Richfield") %>%
  ggplot(aes(x = date, y = hour, fill = count)) +
  geom_tile() +
  scale_fill_viridis_c(limits = c(0, max(pb(data_count$count))),
                       breaks = pb(data_count$count)) +
  #scale_fill_gradient(low = "lightblue", high = "red") +
  labs(x = "Date", y = "Time of Day (Hourly Bins)", fill = "Count") +
  theme_minimal() +
  scale_y_continuous(limits = c(23.5, 0), expand = c(0, 0), breaks = h_breaks,
    trans = "reverse",
    labels = sapply(seq(0, 24, by = 1),
                    function(x) {
                      if (x %% 2 == 0) {
                        sprintf("%02d:00", x)  # Show label for even hours
                      } else {
                        ""  # Leave odd hours blank
                      }
                    })
  ) +  # Set y-axis from 0 to 23 with 2-hour intervals
  scale_x_date(date_labels = "%d %b %Y",
               date_breaks = "2 days",
               expand = c(0, 0)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_grid(cols = vars(recorder), scales = "free_x") +
  #coord_cartesian(xlim = c(data_count$min_date, data_count$max_date)) +
  theme(panel.border = element_rect(fill = 0, linewidth = 1.5),
    axis.text = element_text(size = 20),
    axis.title = element_blank(),
    axis.ticks = element_line(size = 1),
    strip.text.x = element_text(size = 28),
    legend.title = element_text(size = 18),
    legend.key.size = unit(1.5, "cm"),
    legend.text = element_text(size = 15)
  )
