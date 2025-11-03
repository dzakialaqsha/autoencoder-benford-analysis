library(gganimate)
library(ggplot2)
library(dplyr)
library(tidyverse)

# Visualisasi Autoencoder
path <- c(
  '/input_data.csv',
  '/latent_data.csv',
  '/reconstructed_data.csv'
)

list_data <- list()
for(i in 1:length(path)){
  list_data[[i]] <- data.frame(read_csv(path[i], col_names=TRUE))
  list_data[[i]]['time'] <- i
  list_data[[i]]['time'] <- as.vector(list_data[[i]]['time'])
  
  if(i < length(path)){
    list_data[[i]]['predicted_anomaly'] = 0}
}

colnames(list_data[[3]]) <-  c('x', 'y', 'z', 'mse', 'predicted_anomaly', 'anomaly', 'time')

list_data[[4]] <- list_data[[3]]
list_data[[4]]['time'] <- 4
list_data[[4]]['anomaly'] <- list_data[[3]]['predicted_anomaly']
list_data[[4]][which(list_data[[4]]['anomaly'] == 1),'anomaly'] <- 2

shared_cols <- Reduce(intersect, lapply(list_data, names))

combined_df <- dplyr::bind_rows(list_data[[1]][shared_cols], list_data[[2]][shared_cols], list_data[[3]][shared_cols], list_data[[4]][shared_cols])
combined_df$anomaly <- as.factor(combined_df$anomaly)
combined_df$time <- as.numeric(combined_df$time)

###
color_map <- c(
  "0" = "black", # Must be character strings matching factor levels
  "1" = "blue",   
  "2" = "red"     
)

p <- ggplot(
  combined_df,
  aes(x = x, y=y, size=z, colour=anomaly)
) +
  geom_point(show.legend = TRUE, alpha = 0.7) +
  scale_colour_manual(values = color_map, 
                      labels = c("0" = "Normal", "1" = "Actual Anomaly", "2" = "Predicted Anomaly")) +
  scale_size(range = c(2, 12)) +
  labs(
    x = "X Coordinate",
    y = "Y Coordinate",
    color = "Anomaly Status",
    size = "Z Coordinate",
    title = "Autoencoder Anomaly Detection",
    subtitle = "Data transformation throughout the stages"
  ) +
  theme_minimal()+
  transition_time(time) +
  labs(title = "Autoencoder Stage: {as.integer(frame_time)}")+
  view_follow(fixed_x = TRUE, fixed_y = TRUE)+
  shadow_mark(
    data = combined_df %>% filter(time == 1),
    colour = 'gray50',  
    alpha = 0.3,
    size = 4,
    show.legend = FALSE,
    fixed = TRUE
  )

fps_value <- 15
end_pause_frames <- 5 * fps_value 
animate(
  p, 
  fps = fps_value, 
  end_pause = end_pause_frames,
  duration = 15 
)

################################################################################
# New Visualization Regime #
################################################################################
mse_values <- list_data[[3]]$mse
target_min <- min(list_data[[3]]$x)
target_max <- max(list_data[[3]]$x)
mse_min <- min(mse_values)
mse_max <- max(mse_values)

if (mse_max == mse_min) {
  scaled_mse <- rep(target_min, length(mse_values))
} else {
  scaled_mse <- target_min + (
    (mse_values - mse_min) * (target_max - target_min) / (mse_max - mse_min)
  )
}


list_data[[4]] <- list_data[[3]]
list_data[[4]]['x'] <- scaled_mse
list_data[[4]]['y'] <- 0
list_data[[4]]['z'] <- 3
list_data[[4]]['time'] <- 4
list_data[[4]]['anomaly'] <- list_data[[1]]['anomaly']

list_data[[5]] <- list_data[[4]]
list_data[[5]]['anomaly'] <- list_data[[3]]['predicted_anomaly']
list_data[[5]][which(list_data[[5]]['anomaly'] == 1),'anomaly'] <- 2
list_data[[5]]['time'] <- 5

list_data[[6]] <- list_data[[3]]
list_data[[6]]['time'] <- 6
list_data[[6]]['anomaly'] <- list_data[[3]]['predicted_anomaly']
list_data[[6]][which(list_data[[6]]['anomaly'] == 1),'anomaly'] <- 2

shared_cols <- Reduce(intersect, lapply(list_data, names))

combined_df <- dplyr::bind_rows(
  list_data[[1]][shared_cols],
  list_data[[2]][shared_cols], 
  list_data[[3]][shared_cols],
  list_data[[4]][shared_cols],
  list_data[[5]][shared_cols],
  list_data[[6]][shared_cols])

combined_df$anomaly <- as.factor(combined_df$anomaly)
combined_df$time <- as.numeric(combined_df$time)

p <- ggplot(
  combined_df,
  aes(x = x, y=y, size=z, colour=anomaly)
) +
  geom_point(show.legend = TRUE, alpha = 0.7) +
  scale_colour_manual(values = color_map, 
                      labels = c("0" = "Normal", "1" = "Actual Anomaly", "2" = "Predicted Anomaly")) +
  scale_size(range = c(2, 12)) +
  labs(
    x = "X Coordinate",
    y = "Y Coordinate",
    color = "Anomaly Status",
    size = "Z Coordinate",
    title = "Autoencoder Anomaly Detection",
    subtitle = "Data transformation throughout the stages"
  ) +
  theme_minimal()+
  transition_time(time) +
  labs(title = "Autoencoder Stage: {as.integer(frame_time)}")+
  view_follow(fixed_x = TRUE, fixed_y = TRUE)+
  shadow_mark(
    data = combined_df %>% filter(time == 1),
    colour = 'gray50',  
    alpha = 0.3,
    size = 4,
    show.legend = FALSE,
    fixed = TRUE
  )

fps_value <- 15
end_pause_frames <- 5 * fps_value 
animate(
  p, 
  fps = fps_value, 
  end_pause = end_pause_frames,
  duration = 30
)

