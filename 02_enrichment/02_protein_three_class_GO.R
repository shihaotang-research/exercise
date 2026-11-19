# GO BP enrichment across Baseline-only, Dual-response and Dynamic-only proteins.
# Uses the simplified 2,039-protein GO reference (Wang cutoff 0.6).
# Enrichment rows are retained at raw P < 0.05 with BH-FDR provided.
# Baseline direction: median(logFCAD); response direction:
# median(|AUC_Recovery_W8| - |AUC_Recovery_W0|).

#
#   1. protein_baseline_only.tsv
#   2. protein_dual_response.tsv
#   3. protein_dynamic_only.tsv
#
#   go_term_to_gene.tsv
#   go_term_names.tsv
#
#
#   Baseline-only:
#       median(logFCAD) > 0  -> Up
#       median(logFCAD) < 0  -> Down
#
#   Trend-only:
#       deltaAUC = |AUC_Recovery_W8| - |AUC_Recovery_W0|
#       median(deltaAUC) > 0 -> Amplified
#       median(deltaAUC) < 0 -> Attenuated
#
#   Dual-response:
#
#   protein_group_go_enrichment.tsv

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(dplyr)
})

PROJECT <- Sys.getenv("MULTIOMICS_DATA_DIR", unset = file.path(getwd(), "data"))
file_baseline <- file.path(PROJECT, "protein_baseline_only.tsv")
file_both     <- file.path(PROJECT, "protein_dual_response.tsv")
file_trend    <- file.path(PROJECT, "protein_dynamic_only.tsv")

ref_file <- file.path(PROJECT, "protein_annotations.tsv")
bg_file  <- file.path(PROJECT, "protein_background_genes.txt")

term2gene_file <- file.path(PROJECT, "go_term_to_gene.tsv")
term2name_file <- file.path(PROJECT, "go_term_names.tsv")

fc_file      <- file.path(PROJECT, "protein_fold_changes.tsv")
feature_file <- file.path(PROJECT, "molecular_response_features.tsv")

out_file <- file.path(PROJECT, "protein_group_go_enrichment.tsv")

p_cutoff <- 0.05

min_gs_size <- 2

check_file <- function(file) {
  if (!file.exists(file)) {
    stop("Input file not found: ", file)
  }
}

read_protein_list <- function(file) {

  check_file(file)

  df <- read.table(
    file,
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )

  if (ncol(df) < 1) {
    stop("proteins list has no valid columns: ", file)
  }

  out <- data.frame(
    Protein = trimws(as.character(df[[1]])),
    stringsAsFactors = FALSE
  )

  out <- out %>%
    filter(
      !is.na(Protein),
      Protein != ""
    ) %>%
    distinct(Protein, .keep_all = TRUE)

  return(out)
}

load_custom_database <- function(file, required_cols) {

  check_file(file)

  df <- read.table(
    file,
    sep = "\t",
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = "",
    fill = TRUE
  )

  colnames(df) <- gsub("^\ufeff", "", colnames(df))
  colnames(df) <- trimws(colnames(df))

  matched_cols <- vapply(
    required_cols,
    function(target_col) {

      idx <- which(
        tolower(colnames(df)) == tolower(target_col)
      )

      if (length(idx) == 0) {
        return(NA_character_)
      }

      colnames(df)[idx[1]]
    },
    character(1)
  )

  if (any(is.na(matched_cols))) {
    stop(
      "file ",
      file,
      " in Missing required columns: ",
      paste(
        required_cols[is.na(matched_cols)],
        collapse = ", "
      )
    )
  }

  df <- df[, matched_cols, drop = FALSE]
  colnames(df) <- toupper(required_cols)

  for (cc in colnames(df)) {
    df[[cc]] <- trimws(as.character(df[[cc]]))
  }

  df <- df %>%
    filter(
      if_all(
        everything(),
        ~ !is.na(.x) & .x != ""
      )
    ) %>%
    distinct()

  return(df)
}

safe_median <- function(x) {

  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]

  if (length(x) == 0) {
    return(NA_real_)
  }

  median(x)
}

get_term_direction <- function(
    enrichment_df,
    direction_tbl,
    mode = c("baseline", "trend", "both")
) {

  mode <- match.arg(mode)

  if (nrow(enrichment_df) == 0) {
    return(data.frame())
  }

  result_list <- lapply(
    seq_len(nrow(enrichment_df)),
    function(i) {

      current_row <- enrichment_df[i, , drop = FALSE]

      gene_string <- as.character(
        current_row$geneID[[1]]
      )

      gene_symbols <- unlist(
        strsplit(gene_string, "/", fixed = TRUE)
      )

      gene_symbols <- trimws(gene_symbols)
      gene_symbols <- gene_symbols[
        !is.na(gene_symbols) &
          gene_symbols != ""
      ]

      hit_tbl <- direction_tbl %>%
        filter(Symbol %in% gene_symbols)

      med_logFCAD <- safe_median(
        hit_tbl$logFCAD
      )

      med_deltaAUC <- safe_median(
        hit_tbl$deltaAUC
      )

      baseline_direction <- case_when(
        is.na(med_logFCAD)  ~ NA_character_,
        med_logFCAD > 0     ~ "Up",
        med_logFCAD < 0     ~ "Down",
        TRUE                ~ "Neutral"
      )

      trend_direction <- case_when(
        is.na(med_deltaAUC) ~ NA_character_,
        med_deltaAUC > 0    ~ "Amplified",
        med_deltaAUC < 0    ~ "Attenuated",
        TRUE                ~ "Neutral"
      )

      combined_direction <- if (
        !is.na(baseline_direction) &&
          !is.na(trend_direction)
      ) {
        paste0(
          baseline_direction,
          "+",
          trend_direction
        )
      } else {
        NA_character_
      }

      if (mode == "baseline") {

        direction_value <- med_logFCAD
        direction_label <- baseline_direction

      } else if (mode == "trend") {

        direction_value <- med_deltaAUC
        direction_label <- trend_direction

      } else {

        direction_value <- NA_real_
        direction_label <- combined_direction
      }

      data.frame(
        Cluster = as.character(
          current_row$Cluster[[1]]
        ),
        ID = as.character(
          current_row$ID[[1]]
        ),
        Description = as.character(
          current_row$Description[[1]]
        ),

        hit_n_for_direction = nrow(hit_tbl),

        median_logFCAD = med_logFCAD,
        median_deltaAUC = med_deltaAUC,

        baseline_direction = baseline_direction,
        trend_direction = trend_direction,
        combined_direction = combined_direction,

        direction_value = direction_value,
        direction_label = direction_label,

        stringsAsFactors = FALSE
      )
    }
  )

  bind_rows(result_list)
}

make_empty_result <- function() {

  data.frame(
    Cluster = character(),
    ID = character(),
    Description = character(),
    GeneRatio = character(),
    BgRatio = character(),
    pvalue = numeric(),
    FDR = numeric(),
    qvalue = numeric(),
    geneID = character(),
    Count = integer(),
    hit_n_for_direction = integer(),
    median_logFCAD = numeric(),
    median_deltaAUC = numeric(),
    baseline_direction = character(),
    trend_direction = character(),
    combined_direction = character(),
    direction_value = numeric(),
    direction_label = character(),
    stringsAsFactors = FALSE
  )
}

main <- function() {

  input_files <- c(
    file_baseline,
    file_both,
    file_trend,
    ref_file,
    bg_file,
    term2gene_file,
    term2name_file,
    fc_file,
    feature_file
  )

  invisible(
    lapply(input_files, check_file)
  )

  TERM2GENE <- load_custom_database(
    term2gene_file,
    c("term", "gene")
  )

  TERM2NAME <- load_custom_database(
    term2name_file,
    c("term", "name")
  )

  background_genes <- read.table(
    bg_file,
    header = FALSE,
    sep = "\t",
    stringsAsFactors = FALSE,
    quote = "",
    comment.char = "",
    fill = TRUE
  )[[1]]

  background_genes <- trimws(
    as.character(background_genes)
  )

  background_genes <- unique(
    background_genes[
      !is.na(background_genes) &
        background_genes != ""
    ]
  )

  effective_background <- intersect(
    background_genes,
    unique(TERM2GENE$GENE)
  )

  if (length(effective_background) == 0) {
    stop(
      "Background genes in protein_background_genes.txt do not match ",
      "go_term_to_gene.tsv matching. "
    )
  }

  ref_map <- read.table(
    ref_file,
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE,
    fill = TRUE,
    quote = "",
    comment.char = ""
  )

  if (ncol(ref_map) < 2) {
    stop(
      "protein_annotations.tsv must have at least two columns: Protein and Symbol. "
    )
  }

  ref_map <- ref_map[, 1:2, drop = FALSE]
  colnames(ref_map) <- c("Protein", "Symbol")

  ref_map <- ref_map %>%
    mutate(
      Protein = trimws(as.character(Protein)),
      Symbol = trimws(as.character(Symbol))
    ) %>%
    filter(
      !is.na(Protein),
      Protein != "",
      !is.na(Symbol),
      Symbol != ""
    ) %>%
    distinct(
      Protein,
      .keep_all = TRUE
    )

  baseline_df <- read_protein_list(
    file_baseline
  )

  both_df <- read_protein_list(
    file_both
  )

  trend_df <- read_protein_list(
    file_trend
  )

  baseline_map <- baseline_df %>%
    left_join(
      ref_map,
      by = "Protein"
    ) %>%
    filter(
      !is.na(Symbol),
      Symbol != ""
    ) %>%
    distinct(
      Protein,
      Symbol
    )

  both_map <- both_df %>%
    left_join(
      ref_map,
      by = "Protein"
    ) %>%
    filter(
      !is.na(Symbol),
      Symbol != ""
    ) %>%
    distinct(
      Protein,
      Symbol
    )

  trend_map <- trend_df %>%
    left_join(
      ref_map,
      by = "Protein"
    ) %>%
    filter(
      !is.na(Symbol),
      Symbol != ""
    ) %>%
    distinct(
      Protein,
      Symbol
    )

  baseline_gene <- intersect(
    unique(baseline_map$Symbol),
    effective_background
  )

  both_gene <- intersect(
    unique(both_map$Symbol),
    effective_background
  )

  trend_gene <- intersect(
    unique(trend_map$Symbol),
    effective_background
  )

  if (
    length(baseline_gene) == 0 &&
      length(both_gene) == 0 &&
      length(trend_gene) == 0
  ) {
    stop(
      "No class proteins map to the GO reference background. "
    )
  }

  fc_df <- read.table(
    fc_file,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )

  colnames(fc_df) <- trimws(
    colnames(fc_df)
  )

  required_fc_cols <- c(
    "protein",
    "logFCAD"
  )

  missing_fc_cols <- setdiff(
    required_fc_cols,
    colnames(fc_df)
  )

  if (length(missing_fc_cols) > 0) {
    stop(
      "protein_fold_changes.tsv Missing columns: ",
      paste(
        missing_fc_cols,
        collapse = ", "
      )
    )
  }

  fc_df <- fc_df %>%
    transmute(
      Protein = trimws(
        as.character(protein)
      ),
      logFCAD = suppressWarnings(
        as.numeric(logFCAD)
      )
    ) %>%
    filter(
      !is.na(Protein),
      Protein != ""
    ) %>%
    distinct(
      Protein,
      .keep_all = TRUE
    )

  feature_df <- read.table(
    feature_file,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = ""
  )

  colnames(feature_df) <- trimws(
    colnames(feature_df)
  )

  if ("type" %in% colnames(feature_df)) {

    feature_df <- feature_df %>%
      filter(
        tolower(trimws(as.character(type))) ==
          "protein"
      )
  }

  required_feature_cols <- c(
    "molecule",
    "AUC_Recovery_W0",
    "AUC_Recovery_W8"
  )

  missing_feature_cols <- setdiff(
    required_feature_cols,
    colnames(feature_df)
  )

  if (length(missing_feature_cols) > 0) {
    stop(
      "molecular_response_features.tsv Missing columns: ",
      paste(
        missing_feature_cols,
        collapse = ", "
      )
    )
  }

  feature_df <- feature_df %>%
    transmute(
      Protein = trimws(
        as.character(molecule)
      ),

      AUC_Recovery_W0 = suppressWarnings(
        as.numeric(AUC_Recovery_W0)
      ),

      AUC_Recovery_W8 = suppressWarnings(
        as.numeric(AUC_Recovery_W8)
      ),

      deltaAUC =
        abs(
          suppressWarnings(
            as.numeric(AUC_Recovery_W8)
          )
        ) -
        abs(
          suppressWarnings(
            as.numeric(AUC_Recovery_W0)
          )
        )
    ) %>%
    filter(
      !is.na(Protein),
      Protein != ""
    ) %>%
    distinct(
      Protein,
      .keep_all = TRUE
    )

  baseline_direction_tbl <- baseline_map %>%
    left_join(
      fc_df,
      by = "Protein"
    ) %>%
    left_join(
      feature_df,
      by = "Protein"
    ) %>%
    filter(
      Symbol %in% effective_background
    )

  both_direction_tbl <- both_map %>%
    left_join(
      fc_df,
      by = "Protein"
    ) %>%
    left_join(
      feature_df,
      by = "Protein"
    ) %>%
    filter(
      Symbol %in% effective_background
    )

  trend_direction_tbl <- trend_map %>%
    left_join(
      fc_df,
      by = "Protein"
    ) %>%
    left_join(
      feature_df,
      by = "Protein"
    ) %>%
    filter(
      Symbol %in% effective_background
    )

  gene_clusters <- list(
    "Baseline Only" = baseline_gene,
    "Dual-response" = both_gene,
    "Trend Only" = trend_gene
  )

  gene_clusters <- gene_clusters[
    lengths(gene_clusters) > 0
  ]

  compare_result <- compareCluster(
    geneCluster = gene_clusters,
    fun = "enricher",

    universe = effective_background,

    TERM2GENE = TERM2GENE,
    TERM2NAME = TERM2NAME,

    pvalueCutoff = 1,
    pAdjustMethod = "BH",
    qvalueCutoff = 1,

    minGSSize = min_gs_size
  )

  if (
    is.null(compare_result) ||
      nrow(as.data.frame(compare_result)) == 0
  ) {

    empty_result <- make_empty_result()

    write.table(
      empty_result,
      out_file,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE,
      na = "NA"
    )

    return(invisible(NULL))
  }

  enrichment_df <- as.data.frame(
    compare_result
  )

  enrichment_df <- enrichment_df %>%
    filter(
      !is.na(pvalue),
      pvalue < p_cutoff
    )

  if (nrow(enrichment_df) == 0) {

    empty_result <- make_empty_result()

    write.table(
      empty_result,
      out_file,
      sep = "\t",
      row.names = FALSE,
      quote = FALSE,
      na = "NA"
    )

    return(invisible(NULL))
  }

  baseline_result <- enrichment_df %>%
    filter(
      Cluster == "Baseline Only"
    )

  both_result <- enrichment_df %>%
    filter(
      Cluster == "Dual-response"
    )

  trend_result <- enrichment_df %>%
    filter(
      Cluster == "Trend Only"
    )

  baseline_direction <- get_term_direction(
    enrichment_df = baseline_result,
    direction_tbl = baseline_direction_tbl,
    mode = "baseline"
  )

  both_direction <- get_term_direction(
    enrichment_df = both_result,
    direction_tbl = both_direction_tbl,
    mode = "both"
  )

  trend_direction <- get_term_direction(
    enrichment_df = trend_result,
    direction_tbl = trend_direction_tbl,
    mode = "trend"
  )

  direction_df <- bind_rows(
    baseline_direction,
    both_direction,
    trend_direction
  )

  final_result <- enrichment_df %>%
    left_join(
      direction_df,
      by = c(
        "Cluster",
        "ID",
        "Description"
      )
    )

  if (!"p.adjust" %in% colnames(final_result)) {
    final_result$p.adjust <- NA_real_
  }

  final_result <- final_result %>%
    mutate(
      pvalue = suppressWarnings(
        as.numeric(pvalue)
      ),

      FDR = suppressWarnings(
        as.numeric(p.adjust)
      ),

      qvalue = if ("qvalue" %in% colnames(.)) {
        suppressWarnings(
          as.numeric(qvalue)
        )
      } else {
        NA_real_
      }
    ) %>%
    select(
      -p.adjust
    )

  cluster_order <- c(
    "Baseline Only",
    "Dual-response",
    "Trend Only"
  )

  final_result <- final_result %>%
    mutate(
      cluster_order_temp = match(
        Cluster,
        cluster_order
      )
    ) %>%
    arrange(
      cluster_order_temp,
      pvalue,
      FDR
    ) %>%
    select(
      -cluster_order_temp
    )

  preferred_columns <- c(
    "Cluster",
    "ID",
    "Description",

    "GeneRatio",
    "BgRatio",
    "RichFactor",
    "FoldEnrichment",
    "zScore",

    "pvalue",
    "FDR",
    "qvalue",

    "geneID",
    "Count",

    "hit_n_for_direction",

    "median_logFCAD",
    "median_deltaAUC",

    "baseline_direction",
    "trend_direction",
    "combined_direction",

    "direction_value",
    "direction_label"
  )

  final_result <- final_result %>%
    select(
      any_of(preferred_columns),
      everything()
    )

  write.table(
    final_result,
    out_file,
    sep = "\t",
    row.names = FALSE,
    quote = FALSE,
    na = "NA"
  )

  result_summary <- final_result %>%
    count(
      Cluster,
      name = "enriched_term_n"
    )

  for (i in seq_len(nrow(result_summary))) {
  }

}

main()
message("Three-class protein enrichment results saved.")
