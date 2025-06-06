readRenviron("/etc/default/locale")
LANG <- Sys.getenv("LANG")
if(nchar(LANG))
   Sys.setlocale("LC_ALL", LANG)

install.packages("pak")
pak::pkg_install(c("BiocManager", "devtools"))
options(pak.extra_repos = c(Bioc = "https://bioconductor.org/packages/3.18/bioc"))
pak::install(c("DEP", "SummarizedExperiment", "MSnbase", "pcaMethods", "vsn", "impute"))
pak::pkg_install(c("IRkernel", "tidyverse", "flashClust", "proteomicsCV", "samr", "WGCNA", "imputeLCMD"))

library(devtools)
install_github("https://github.com/vdemichev/diann-rpackage")