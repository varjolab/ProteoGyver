library(diann)
df <- diann_load("report.tsv")
df$modified <- grepl("UniMod:21", df$Modified.Sequence, fixed=TRUE)
df <- df[df$modified==TRUE,]
df$File.Name <- df$Run
mod_pgs <- diann_maxlfq(df, group.header="Protein.Group", id.header = "Precursor.Id", quantity.header = "Precursor.Normalised")
write.table(
  data.frame(Protein.Group = rownames(mod_pgs), mod_pgs),
  "matrix_phospho2.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
