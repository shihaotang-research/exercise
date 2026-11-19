#!/usr/bin/env Rscript
# 01. Protein (2039) + retained metabolites (1142): mixed models with beta
# Model: value ~ group * Time + sex + age + (1 | ID)
# Test: Type III Satterthwaite F test of group:Time (3 df)
# FDR: BH over all successfully tested molecules, once (not separately by omics)
# Additional descriptive effects: W8-vs-W0 differences in acute changes at
#   2, 30, and 60 min, relative to the respective pre-exercise baseline.
# Existing results are NOT overwritten; optional strict old-vs-new check.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(parallel)
})

# Data directory may be outside the public repository.
# Set MULTIOMICS_DATA_DIR for server-specific locations. If unset, run
# Rscript from the repository root with input files under ./data.
PROJECT <- Sys.getenv("MULTIOMICS_DATA_DIR", unset = file.path(getwd(), "data"))
INPUT_DIR <- file.path(PROJECT, "lmm_input")
MET_LIST <- file.path(PROJECT, "metabolite_ids.txt")
OLD_RESULTS <- file.path(PROJECT, "interaction_lmm_results.tsv")
OUT_DIR <- file.path(PROJECT, "lmm_interaction_effects")
N_CORES <- 10L
CHECK_OLD_RESULTS <- TRUE    # Stop if historical P/FDR values change unexpectedly.
ALLOW_DF_DF1_TIME <- TRUE     # Recode DF: D/F1/F2/F3 -> A/B1/B2/B3, if necessary.

if (!dir.exists(INPUT_DIR)) stop("Input directory not found: ", INPUT_DIR)
if (!file.exists(MET_LIST)) stop("Metabolite list not found: ", MET_LIST)

met_keep <- unique(trimws(scan(MET_LIST, what = "character", quiet = TRUE)))
met_keep <- met_keep[nzchar(met_keep)]
all_files <- list.files(INPUT_DIR, pattern = "\\.txt$", full.names = TRUE)
file_info <- data.frame(
  FilePath = all_files,
  File = basename(all_files),
  Molecule = tools::file_path_sans_ext(basename(all_files)),
  stringsAsFactors = FALSE
)
file_info$Omics <- ifelse(grepl("^met", file_info$Molecule, ignore.case = TRUE),
                          "Metabolite", "Protein")
file_info <- file_info[
  file_info$Omics == "Protein" |
    (file_info$Omics == "Metabolite" & file_info$Molecule %in% met_keep),
  , drop = FALSE
]
file_info <- file_info[order(file_info$File), , drop = FALSE]
row.names(file_info) <- NULL

n_pro <- sum(file_info$Omics == "Protein")
n_met <- sum(file_info$Omics == "Metabolite")
cat(sprintf("Files: %d protein + %d metabolite = %d total\n", n_pro, n_met, nrow(file_info)))
if (n_pro != 2039L || n_met != 1142L || length(met_keep) != 1142L) {
  stop("Expected exactly 2039 proteins and 1142 metabolites. Review input files and metabolite_ids.txt.")
}
if (anyDuplicated(file_info$File) || anyDuplicated(met_keep)) stop("Duplicate molecule filename/list entry")

# Return the exact same columns for successes and errors.
empty_row <- function(i, status, singular = NA, n_obs = NA_integer_, n_ID = NA_integer_) {
  data.frame(
    File = file_info$File[i], Omics = file_info$Omics[i],
    p_interaction = NA_real_, singular = singular, status = status,
    F_interaction = NA_real_, NumDF = NA_real_, DenDF = NA_real_,
    beta_2min = NA_real_, beta_30min = NA_real_, beta_60min = NA_real_,
    SE_2min = NA_real_, SE_30min = NA_real_, SE_60min = NA_real_,
    n_obs = n_obs, n_ID = n_ID, stringsAsFactors = FALSE
  )
}

# The original analysis assumes identical encoded Time levels in both groups.
# Some raw files may instead use DF time names D/F1/F2/F3. If so, harmonize
# ONLY those DF rows; reject mixed labeling rather than silently excluding data.
prepare_data <- function(dat) {
  required <- c("ID", "group", "Time", "value", "sex", "age")
  if (!all(required %in% names(dat))) return(list(error = "missing_columns"))
  dat <- dat[, required, drop = FALSE]
  dat$ID <- trimws(as.character(dat$ID))
  dat$group <- trimws(as.character(dat$group))
  dat$Time <- trimws(as.character(dat$Time))
  dat <- dat[dat$ID != "", , drop = FALSE]

  w0_time <- c("A", "B1", "B2", "B3")
  w8_time <- c("D", "F1", "F2", "F3")
  df_times <- unique(dat$Time[dat$group == "DF"])
  if (any(df_times %in% w8_time)) {
    if (!ALLOW_DF_DF1_TIME) return(list(error = "DF_time_needs_harmonization"))
    if (any(df_times %in% w0_time)) return(list(error = "DF_mixed_time_coding"))
    idx <- which(dat$group == "DF" & dat$Time %in% w8_time)
    dat$Time[idx] <- w0_time[match(dat$Time[idx], w8_time)]
  }
  if (any(dat$group == "AB" & dat$Time %in% w8_time)) {
    return(list(error = "AB_unexpected_DF_times"))
  }
  # Ignore extra non-target timepoints only if they do not indicate a malformed
  # group design; the modeling timepoints must form an unambiguous 2x4 table.
  dat <- dat[dat$group %in% c("AB", "DF") & dat$Time %in% w0_time,
             , drop = FALSE]
  dat$value <- suppressWarnings(as.numeric(dat$value))
  dat$age <- suppressWarnings(as.numeric(dat$age))
  dat <- dat[stats::complete.cases(dat), , drop = FALSE]
  if (nrow(dat) == 0L) return(list(error = "no_data"))

  dat$ID <- factor(dat$ID)
  dat$group <- factor(dat$group, levels = c("AB", "DF"))
  dat$Time <- factor(dat$Time, levels = w0_time)
  dat$sex <- factor(dat$sex)
  dat <- droplevels(dat)
  if (nlevels(dat$group) != 2L || nlevels(dat$Time) != 4L || nlevels(dat$sex) < 2L) {
    return(list(error = "invalid_design"))
  }
  counts <- table(dat$group, dat$Time)
  if (!all(dim(counts) == c(2L, 4L)) || any(counts == 0L)) {
    return(list(error = "incomplete_2x4_design"))
  }
  # Explicit treatment contrasts give interpretable beta names.
  contrasts(dat$group) <- contr.treatment(levels(dat$group), base = 1L)
  contrasts(dat$Time) <- contr.treatment(levels(dat$Time), base = 1L)
  list(data = dat, counts = counts)
}

# Print an actual pre-fit cross-tab for one input file. Check DF time encoding.
example <- tryCatch(read.delim(file_info$FilePath[1], check.names = FALSE,
                               stringsAsFactors = FALSE), error = function(e) NULL)
if (!is.null(example) && all(c("group", "Time") %in% names(example))) {
  cat("\nFirst file - raw group x Time:\n")
  print(table(example$group, example$Time))
  example_clean <- prepare_data(example)
  if (!is.null(example_clean$data)) {
    cat("First file - modeled group x Time:\n")
    print(example_clean$counts)
  } else {
    cat("First file data-check status:", example_clean$error, "\n")
  }
}

analyze_one <- function(i) {
  raw <- tryCatch(read.delim(file_info$FilePath[i],
                             stringsAsFactors = FALSE, check.names = FALSE),
                  error = function(e) NULL)
  if (is.null(raw)) return(empty_row(i, "read_error"))
  prepared <- prepare_data(raw)
  if (!is.null(prepared$error)) return(empty_row(i, prepared$error))
  dat <- prepared$data
  n_obs <- nrow(dat)
  n_ID <- nlevels(dat$ID)

  fit <- tryCatch(
    suppressWarnings(lmerTest::lmer(
      value ~ group * Time + sex + age + (1 | ID), data = dat, REML = TRUE,
      control = lmerControl(optimizer = "nloptwrap", optCtrl = list(maxeval = 20000))
    )), error = function(e) NULL
  )
  if (is.null(fit)) return(empty_row(i, "model_error", n_obs = n_obs, n_ID = n_ID))
  singular <- lme4::isSingular(fit, tol = 1e-4)

  a <- tryCatch(suppressWarnings(anova(fit, type = 3, ddf = "Satterthwaite")),
                error = function(e) NULL)
  if (is.null(a) || !("group:Time" %in% rownames(a))) {
    return(empty_row(i, "anova_error", singular, n_obs, n_ID))
  }

  b <- lme4::fixef(fit)
  v <- tryCatch(diag(as.matrix(vcov(fit))), error = function(e) NULL)
  keys <- c("groupDF:TimeB1", "groupDF:TimeB2", "groupDF:TimeB3")
  if (!all(keys %in% names(b)) || is.null(v) || !all(keys %in% names(v))) {
    return(empty_row(i, "beta_extraction_error", singular, n_obs, n_ID))
  }
  p <- as.numeric(a["group:Time", "Pr(>F)"])
  f_stat <- as.numeric(a["group:Time", "F value"])
  if (!is.finite(p) || !is.finite(f_stat)) {
    return(empty_row(i, "invalid_F_test", singular, n_obs, n_ID))
  }

  data.frame(
    File = file_info$File[i], Omics = file_info$Omics[i],
    p_interaction = p, singular = singular, status = "success",
    F_interaction = f_stat,
    NumDF = as.numeric(a["group:Time", "NumDF"]),
    DenDF = as.numeric(a["group:Time", "DenDF"]),
    beta_2min = unname(b[keys[1]]),
    beta_30min = unname(b[keys[2]]),
    beta_60min = unname(b[keys[3]]),
    SE_2min = sqrt(unname(v[keys[1]])),
    SE_30min = sqrt(unname(v[keys[2]])),
    SE_60min = sqrt(unname(v[keys[3]])),
    n_obs = n_obs, n_ID = n_ID, stringsAsFactors = FALSE
  )
}

cat("\nFitting models ...\n")
res_list <- parallel::mclapply(seq_len(nrow(file_info)), analyze_one,
                                mc.cores = min(N_CORES, nrow(file_info)),
                                mc.preschedule = TRUE)
if (any(vapply(res_list, inherits, logical(1), what = "try-error"))) {
  stop("At least one worker crashed. Check input/available memory; nothing written.")
}
res <- do.call(rbind, res_list)
row.names(res) <- NULL
valid <- is.finite(res$p_interaction) & res$status == "success"
res$FDR_all_3181 <- NA_real_
res$FDR_all_3181[valid] <- p.adjust(res$p_interaction[valid], method = "BH")

# Historical parity check before writing ANY result files.
if (CHECK_OLD_RESULTS && file.exists(OLD_RESULTS)) {
  old <- read.delim(OLD_RESULTS, check.names = FALSE, stringsAsFactors = FALSE)
  req <- c("File", "Omics", "p_interaction", "FDR_all_3181")
  if (!all(req %in% names(old))) stop("Historical file missing required columns.")
  key_new <- paste(res$Omics, res$File, sep = "|")
  key_old <- paste(old$Omics, old$File, sep = "|")
  if (anyDuplicated(key_old) || !setequal(key_new, key_old)) {
    stop("Historical and new molecule ID sets differ. Not overwriting anything.")
  }
  old <- old[match(key_new, key_old), , drop = FALSE]
  old_p <- suppressWarnings(as.numeric(old$p_interaction))
  old_q <- suppressWarnings(as.numeric(old$FDR_all_3181))
  mismatch_p <- xor(is.finite(old_p), is.finite(res$p_interaction)) |
    (is.finite(old_p) & is.finite(res$p_interaction) &
       abs(old_p - res$p_interaction) > 1e-9 + 1e-5 * abs(old_p))
  mismatch_q <- xor(is.finite(old_q), is.finite(res$FDR_all_3181)) |
    (is.finite(old_q) & is.finite(res$FDR_all_3181) &
       abs(old_q - res$FDR_all_3181) > 1e-9 + 1e-5 * abs(old_q))
  mismatch_p[is.na(mismatch_p)] <- TRUE
  mismatch_q[is.na(mismatch_q)] <- TRUE
  cat(sprintf("Historical check: p mismatches=%d; FDR mismatches=%d\n",
              sum(mismatch_p), sum(mismatch_q)))
  if (any(mismatch_p | mismatch_q)) {
    ii <- which(mismatch_p | mismatch_q)[1:min(10L, sum(mismatch_p | mismatch_q))]
    print(data.frame(File = res$File[ii], old_p = old_p[ii], new_p = res$p_interaction[ii],
                     old_FDR = old_q[ii], new_FDR = res$FDR_all_3181[ii]))
    stop("New vs old LMM result mismatch; inspect group/Time coding and inputs first.")
  }
} else if (CHECK_OLD_RESULTS) {
  cat("Historical check skipped: original results file absent.\n")
}

if (sum(valid) != 3181L) {
  cat("WARNING: Valid tests = ", sum(valid), ", not 3181. BH used valid tests only.\n", sep = "")
  print(table(res$status))
  stop("At least one molecule was not tested successfully. Resolve before publishing.")
}

sig <- res[!is.na(res$FDR_all_3181) & res$FDR_all_3181 < 0.05, , drop = FALSE]
options(digits = 16)  # Retain high numeric precision in write.table output.
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write_tsv <- function(x, name) {
  write.table(x, file.path(OUT_DIR, name), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}
write_tsv(res, "interaction_lmm_with_effects.tsv")
write_tsv(sig, "interaction_lmm_significant.tsv")
write_tsv(res[res$Omics == "Protein", , drop = FALSE], "protein_interaction_lmm_with_effects.tsv")
write_tsv(res[res$Omics == "Metabolite", , drop = FALSE], "metabolite_interaction_lmm_with_effects.tsv")

cat("\nCompleted. Output directory: ", normalizePath(OUT_DIR), "\n", sep = "")
cat("Successfully tested: ", sum(valid), "/", nrow(res), "\n", sep = "")
cat("Protein FDR < 0.05: ", sum(sig$Omics == "Protein"), "\n", sep = "")
cat("Metabolite FDR < 0.05: ", sum(sig$Omics == "Metabolite"), "\n", sep = "")
cat("Total FDR < 0.05: ", nrow(sig), "\n", sep = "")
cat("Singular fitted models: ", sum(res$singular %in% TRUE), "\n", sep = "")
