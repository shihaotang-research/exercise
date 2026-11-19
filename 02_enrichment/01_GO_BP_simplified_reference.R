# Build one GO BP term universe for the 2,039-protein background.
# Filter GO terms by Count >= 3, apply Wang simplify cutoff 0.6,
# and export the TERM2GENE and TERM2NAME resources.
# Apply the same resource to each RNK file with BH-adjusted enrichment.

library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(AnnotationDbi)
library(readr)
library(stringr)

PROJECT <- Sys.getenv("MULTIOMICS_DATA_DIR", unset = file.path(getwd(), "data"))
rnk_dir <- file.path(PROJECT, "go_ranked_lists")
output_dir <- file.path(PROJECT, "go_enrichment_results")

if (!dir.exists(output_dir)) dir.create(output_dir)

background_genes <- unique(read.table(file.path(PROJECT, "protein_background_genes.txt"), header = FALSE)[,1])

ego_bg <- enrichGO(
  gene          = background_genes,
  OrgDb         = org.Hs.eg.db,
  keyType       = "SYMBOL",
  ont           = "BP",
  pvalueCutoff  = 1,
  qvalueCutoff  = 1,
  readable      = FALSE,
  universe      = background_genes
)

ego_bg_df <- as.data.frame(ego_bg)

valid_terms <- ego_bg_df %>% filter(Count >= 3) %>% pull(ID)
ego_bg@result <- ego_bg@result[ego_bg@result$ID %in% valid_terms, ]

ego_sim_bg <- simplify(
  ego_bg,
  cutoff     = 0.6,
  by         = "p.adjust",
  select_fun = min,
  measure    = "Wang"
)

kept_go_ids <- ego_sim_bg@result$ID

term2gene_raw <- AnnotationDbi::select(
  org.Hs.eg.db,
  keys    = kept_go_ids,
  columns = "SYMBOL",
  keytype = "GOALL"
)

custom_TERM2GENE <- term2gene_raw %>%
  filter(.data[["SYMBOL"]] %in% background_genes) %>%
  transmute(
    Term = .data[["GOALL"]],
    Gene = .data[["SYMBOL"]]
  )

custom_TERM2NAME <- ego_sim_bg@result %>%
  transmute(
    Term = ID,
    Name = Description
  )

write.table(
  custom_TERM2GENE,
  file = file.path(PROJECT, "go_term_to_gene.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

write.table(
  custom_TERM2NAME,
  file = file.path(PROJECT, "go_term_names.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

rnk_files <- list.files(rnk_dir, pattern = "\\.rnk$", full.names = TRUE)

run_unified_GO <- function(rnk_file) {

  base <- tools::file_path_sans_ext(basename(rnk_file))

  df <- read.table(rnk_file, header = FALSE, sep = "\t", stringsAsFactors = FALSE)
  target_genes <- unique(df[,1])

  res <- enricher(
    gene          = target_genes,
    universe      = background_genes,
    TERM2GENE     = custom_TERM2GENE,
    TERM2NAME     = custom_TERM2NAME,
    pvalueCutoff  = 1,
    qvalueCutoff  = 1,
    pAdjustMethod = "BH"
  )

  res_df <- as.data.frame(res)

  if (nrow(res_df) == 0) {
    warning("⚠ input GO term input: ", base)
    return(NULL)
  }

  out_file <- file.path(output_dir, paste0(base, "_go_enrichment.tsv"))

  write.table(res_df, file = out_file, sep = "\t",
              row.names = FALSE, quote = FALSE)

}

lapply(rnk_files, run_unified_GO)
message("GO reference and per-list enrichment results saved.")
