# Script to map an input csv of human gene symbols to the hg38 reference genome.
#
# Author: Jacob Bumgarner, Ph.D.
#
# USAGE:
# 1. Create an input .csv with a single column of gene symbols, titled `Symbol`
# 2. Define the input .csv path in the `SCRIPT INPUTS` section.
# 3. Define an output .csv path in the `SCRIPT INPUTS` section.
# 4. Run script.
#
# OUTPUT:
# - Mapped Symbols: A full list of all transcripts associated with each mapped
#   input gene.
# - Slim Mapped Symbols: A list of the first occurring transcript (lowest
#   chromosomal start position) for each mapped input gene.

#### SCRIPT INPUTS ####
INPUT_SYMBOLS_FILE <- "example_input.csv"
OUTPUT_FILE <- "example_output.csv"
BIN_BP_RESOLUTION <- 5000
#### END SCRIPT INPUTS ####

#### Outline ####
# 0. User script inputs
# 1. Script functions
# 2. Script running
#### End Outline ####

#### Functions ####
#' Load the input csv file.
load_user_symbols <- function(input_symbols_file) {
  if (!file.exists(input_symbols_file)) {
    cli::cli_abort("The input file does not exist: {input_symbols_file}")
  }

  if (tools::file_ext(input_symbols_file) != "csv") {
    cli::cli_abort("The `INPUT_SYMBOLS_FILE` must be a {.field .csv} file.")
  }

  input_symbols <- read.csv(input_symbols_file, header = TRUE) |>
    tibble::as_tibble()

  if (ncol(input_symbols) != 1 ||
      tolower(colnames(input_symbols))[1] != "symbol") {
    cli::cli_abort(
      "The input symbols file should have a single column of human gene symbols, with a column name: {.field Symbol}"
    )
  }

  input_symbols
}


#' Save the output of a mapped symbol file.
#'
#' Saves a full copy and a trimmed version of the output file. The slimmed down
#' output file returns a single transcript per symbol with the earliest
#' chromosomal start position.
save_mapped_symbols <- function(mapped_symbols, out_file) {
  base_dir <- dirname(out_file)
  if (!dir.exists(base_dir)) {
    dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  }

  write.csv(mapped_symbols, out_file, row.names = FALSE)

  # Slim down the mapped symbols
  mapped_symbols_slim <- mapped_symbols |>
    dplyr::group_by(Symbol, Symbol.Alt) |>
    dplyr::slice_min(Start.BP, n = 1, with_ties = FALSE)

  base_out_file <- basename(out_file)
  slim_out_file <- file.path(base_dir, paste0("slim_", base_out_file))
  write.csv(mapped_symbols_slim, slim_out_file, row.names = FALSE)

  invisible()
}

#' Map an input gene to the hg38 reference genome.
map_symbols <- function(input_symbols, bp_bin_width = 5000) {
  # Prepare DBs
  library(org.Hs.eg.db)
  library(TxDb.Hsapiens.UCSC.hg38.knownGene)

  symbols <- AnnotationDbi::select(
    org.Hs.eg.db,
    keys = input_symbols$Symbol,
    columns = c("SYMBOL", "ENSEMBL", "ENTREZID"),
    keytype = "SYMBOL"
  ) |>
    tibble::as_tibble()

  # Warn about unmapped symbols.
  not_mapped <- symbols |>
    dplyr::filter(is.na(ENTREZID))
  symbols <- symbols |>
    dplyr::filter(!is.na(ENTREZID))

  if (nrow(not_mapped)) {
    cli::cli_alert("Attempting to remap {nrow(not_mapped)} symbols using aliases.")

    remapped <- AnnotationDbi::select(
      org.Hs.eg.db,
      not_mapped$SYMBOL,
      columns = c("SYMBOL", "ALIAS", "ENSEMBL", "ENTREZID"),
      keytype = "ALIAS"
    ) |>
      tibble::as_tibble() |>
      dplyr::rename(SYMBOL = ALIAS, SYMBOL.ALT = SYMBOL)

    final_unmapped <- any(is.na(remapped$ENTREZID))

    symbols <- dplyr::bind_rows(symbols, remapped) |>
      dplyr::arrange(SYMBOL)

    if (final_unmapped) {
      cli::cli_warn(
        c(
          "Some of the input gene symbols couldn't be mapped to the {.field org.Hs.eg.db} annotation data base.",
          "i" = "The symbols that failed to map will be in the output file with `NA` chromosome columns.",
          "i" = "The symbols that failed to map will not be included in the slim output file."
        )
      )
    }
  }

  mapped <- AnnotationDbi::select(
    TxDb.Hsapiens.UCSC.hg38.knownGene,
    symbols$ENTREZID,
    columns = c("GENEID", "TXID", "TXCHROM", "TXSTART", "TXEND"),
    keytype = "GENEID"
  )

  mapped <- mapped |>
    dplyr::left_join(symbols,
                     by = c("GENEID" = "ENTREZID"),
                     relationship = "many-to-many") |>
    tibble::as_tibble()

  new_colnames <- c(
    "SYMBOL" = "Symbol",
    "SYMBOL.ALT" = "Symbol.Alt",
    "ENSEMBL" = "Ensembl.ID",
    "GENEID" = "Entrez.ID",
    "TXCHROM" = "Chromosome",
    "TXSTART" = "Start.BP",
    "TXEND" = "End.BP",
    "TXID" = "TxDb.hg38.Transcript.ID"
  )

  colnames(mapped) <- new_colnames[colnames(mapped)]

  # filter out alt chr info
  mapped <- mapped |>
    dplyr::filter(grepl("^chr[0-9XYM]+$", Chromosome)) |>
    dplyr::arrange(Symbol)

  # Add bin resolution
  mapped <- mapped |>
    dplyr::mutate(Bin.ID = Start.BP %/% bp_bin_width)

  mapped
}

#### End Functions ####

#### Run Script ####
input_symbols <- load_user_symbols(input_symbols_file = INPUT_SYMBOLS_FILE)
mapped_symbols <- map_symbols(input_symbols = input_symbols, bp_bin_width = BIN_BP_RESOLUTION)
save_mapped_symbols(mapped_symbols = mapped_symbols, out_file = OUTPUT_FILE)

#### End Run Script ####
