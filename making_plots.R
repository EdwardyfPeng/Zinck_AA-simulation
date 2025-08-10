library(tidyverse)
library(ggplot2)
library(patchwork)

making_plot <- function(data, pos_value, u_value) {
  sumdat <- data %>%
    filter(pos == pos_value, u == u_value) %>%
    group_by(p, target_fdr, method) %>%
    summarise(
      empirical_fdr = mean(empirical_fdr, na.rm = TRUE),
      power         = mean(power,         na.rm = TRUE),
      .groups = "drop"
    ) %>%
    group_by(p, method) %>%
    complete(target_fdr = union(0, target_fdr)) %>%
    mutate(
      empirical_fdr = ifelse(is.na(empirical_fdr) & target_fdr == 0, 0, empirical_fdr),
      power         = ifelse(is.na(power)         & target_fdr == 0, 0, power)
    ) %>%
    ungroup()

  p_fdr <- ggplot(sumdat, aes(x = target_fdr, y = empirical_fdr, color = method)) +
    geom_line() +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "black") +
    facet_wrap(~ p,labeller = labeller(p = function(x) paste0("p=", x))) +
    scale_color_manual(values = c("green", "orange", "blue", "red")) +
    labs(x = NULL, y = "Empirical FDR") +
    theme_bw() +
    theme(legend.position = "bottom")

  p_power <- ggplot(sumdat, aes(x = target_fdr, y = power, color = method)) +
    geom_line() +
    geom_point() +
    facet_wrap(~ p, labeller = labeller(p = function(x) paste0("p=", x))) +
    scale_color_manual(values = c("green", "orange", "blue", "red")) +
    labs(x = "Target FDR", y = "Power") +
    theme_bw() +
    theme(legend.position = "bottom")

  p_fdr / p_power +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
}
