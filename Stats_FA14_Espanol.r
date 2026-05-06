# prerequisites
library(tidyverse) # For data manipulation[cite: 3]
library(haven)     # To read the SPSS .sav file
library(rstatix)   # For summary statistics and ANOVA functions[cite: 3]
library(ggpubr)    # For creating publication-ready plots[cite: 3]

#data preperation
dating_data <- read_sav("Looksorpersonality.sav")
names(dating_data) <- lc_names <- tolower(names(dating_data))

dating_long <- dating_data |>
  mutate(id = row_number()) |>
  pivot_longer(
    cols = -c(gender, id), 
    names_to = "condition",
    values_to = "rating"
  ) |>
  separate(condition, into = c("looks", "personality"), sep = "_") |>
  convert_as_factor(id, gender, looks, personality)

print(head(dating_long))

#summary statistics
summary_stats <- dating_long |>
  group_by(gender, looks, personality) |>
  get_summary_stats(rating, type = "mean_sd")

print(summary_stats)

#visualization
bxp <- ggboxplot(
  dating_long, x = "gender", y = "rating",
  color = "looks", palette = "jco",
  facet.by = "personality", 
  short.panel.labs = FALSE,
  xlab = "Participant Gender", 
  ylab = "Desirability Rating"
)
print(bxp)

#assumption #4: check for outliers
outliers <- dating_long %>%
  group_by(gender, looks, personality) %>%
  identify_outliers(rating)
print(outliers)

#assumption #5: check for normaility
#Shapiro-Wilk test
normality_test <- dating_long %>%
  group_by(gender, looks, personality) %>%
  shapiro_test(rating)
print(normality_test)

#QQ Plots for visual
qq_plot <- ggqqplot(dating_long, "rating", ggtheme = theme_bw()) +
  facet_grid(looks ~ personality, labeller = "label_both")
print(qq_plot)

#assumption #6: homogeneity of variance
levene_results <- dating_long %>%
  group_by(looks, personality) %>%
  levene_test(rating ~ gender)
print(levene_results)

#assumption #7: sphericity
res.aov <- anova_test(
  data = dating_long, 
  dv = rating, 
  wid = id,
  between = gender, 
  within = c(looks, personality)
)
print(get_anova_table(res.aov))

#two-way interaction: looks*personality
two_way_inter <- dating_long %>%
  group_by(gender) %>%
  anova_test(dv = rating, wid = id, within = c(looks, personality))
print(get_anova_table(two_way_inter))

#simple main effect of personality
simple_main <- dating_long %>%
  group_by(gender, looks) %>%
  anova_test(dv = rating, wid = id, within = personality)
print(simple_main %>% get_anova_table() %>% filter(p < .05))


#pairwise comparisons between personality levels
pwc <- dating_long %>%
  group_by(gender, looks) %>%
  pairwise_t_test(
    rating ~ personality, 
    paired = TRUE, 
    p.adjust.method = "bonferroni"
  )

print(pwc)
