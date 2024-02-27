install.packages('BiocManager', 'vsn', 'IRkernel', 'devtools', 'tidyverse', 'flashClust', 'WGCNA', 'samr', ncpus=12)
BiocManager::install(c('DEP', 'AnnotationDb', 'GO.db', 'preprocessCore', 'impute'))
IRkernel::installspec(user = FALSE) 