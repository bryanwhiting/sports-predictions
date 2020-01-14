# Fix the datetime logic in the 

install.load::install_load('tidyverse', 'dplyr', 'caret')
df <- read_csv("data/proc/2020-01-09/mrd.csv")

# Step 1: build a linear regression
# 2: do automl
# 3: do cross validation, hyperparemeter tuning


df %>%
  select(-boxscore_index, -year, -ymdhms, -ends_with('abbr')) %>%
  View()
