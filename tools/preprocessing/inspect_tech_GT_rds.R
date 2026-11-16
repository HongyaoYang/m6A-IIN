#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript inspect_tech_GT_rds.R /path/to/tech_GT")
}
base_dir <- normalizePath(args[[1]], mustWork = TRUE)

split_path <- file.path(base_dir, "split_sites.rds")
if (file.exists(split_path)) {
  split_sites <- readRDS(split_path)
  cat("split_sites.rds class:", paste(class(split_sites), collapse = ","), "\n")
  cat("split names:", paste(names(split_sites), collapse = ", "), "\n")
  for (split_name in names(split_sites)) {
    cat("\n", split_name, "\n", sep = "")
    print(vapply(split_sites[[split_name]], length, integer(1)))
  }
} else {
  cat("Missing split_sites.rds\n")
}

grg_path <- file.path(base_dir, "grg.rds")
if (!file.exists(grg_path)) {
  cat("\nMissing grg.rds\n")
} else if (!requireNamespace("GenomicRanges", quietly = TRUE)) {
  cat("\nCannot inspect grg.rds: R package GenomicRanges is not installed.\n")
  cat("Load the HPC Bioconductor environment and rerun this script.\n")
} else {
  grg <- readRDS(grg_path)
  cat("\ngrg.rds class:", paste(class(grg), collapse = ","), "\n")
  cat("number of genomic ranges:", length(grg), "\n")
  cat("metadata columns:", paste(names(S4Vectors::mcols(grg)), collapse = ", "), "\n")
  cat("sequence levels:", paste(GenomeInfoDb::seqlevels(grg), collapse = ", "), "\n")
  sequence_like <- grep(
    "seq|sequence|window|motif",
    names(S4Vectors::mcols(grg)),
    value = TRUE,
    ignore.case = TRUE
  )
  cat("sequence-like metadata columns:",
      if (length(sequence_like)) paste(sequence_like, collapse = ", ") else "NONE",
      "\n")
  print(utils::head(as.data.frame(grg), 3))
}
