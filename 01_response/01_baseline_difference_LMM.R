library(lme4)
library(lmerTest)

# Run from the repository root or set MULTIOMICS_DATA_DIR.
PROJECT <- Sys.getenv("MULTIOMICS_DATA_DIR", unset = file.path(getwd(), "data"))

met <- read.table(
  file.path(PROJECT, "metabolite_matrix.tsv"),
  sep = "\t",
  header = TRUE,
  check.names = FALSE
)

pro <- read.table(
  file.path(PROJECT, "protein_matrix.tsv"),
  sep = "\t",
  header = TRUE,
  check.names = FALSE
)

met_keep <- scan(
  file.path(PROJECT, "metabolite_ids.txt"),
  what = "character",
  quiet = TRUE
)

met <- met[
  !met$time %in% c("B4", "F4"),
]

pro <- pro[
  !pro$time %in% c("B4", "F4"),
]

stopifnot(
  nrow(met) == nrow(pro),
  identical(
    paste(met$id, met$time),
    paste(pro$id, pro$time)
  )
)

met_keep <- intersect(
  met_keep,
  colnames(met)
)

meta <- c(
  "id",
  "time",
  "fat",
  "sex",
  "age",
  "bmi"
)

pro_cols <- setdiff(
  colnames(pro),
  meta
)

data <- cbind(
  met[, c(meta, met_keep)],
  pro[, pro_cols]
)

data <- data[
  data$time %in% c("A", "D"),
]

data$time <- factor(
  data$time,
  levels = c("A", "D")
)

data$id <- factor(data$id)
data$sex <- factor(data$sex)

molecules <- c(
  met_keep,
  pro_cols
)

results <- data.frame(
  Molecule = molecules,
  p_value = NA_real_
)

for (i in seq_along(molecules)) {

  mol <- molecules[i]

  model <- lmer(
    as.formula(
      paste0(
        "`", mol,
        "` ~ time + sex + age + (1|id)"
      )
    ),
    data = data
  )

  results$p_value[i] <-
    summary(model)$coefficients[
      "timeD",
      "Pr(>|t|)"
    ]
}

results$FDR <- p.adjust(
  results$p_value,
  method = "BH"
)

write.table(
  results,
  file = file.path(PROJECT, "baseline_lmm_results.tsv"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

cat(
  "Subjects:",
  length(unique(data$id)),
  "\n"
)

cat(
  "Metabolites:",
  length(met_keep),
  "\n"
)

cat(
  "Proteins:",
  length(pro_cols),
  "\n"
)

cat(
  "Total molecules:",
  nrow(results),
  "\n"
)

cat(
  "P < 0.05:",
  sum(results$p_value < 0.05, na.rm = TRUE),
  "\n"
)

cat(
  "FDR < 0.05:",
  sum(results$FDR < 0.05, na.rm = TRUE),
  "\n"
)
