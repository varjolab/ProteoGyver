install.packages("pacman")
pacman::p_load('BiocManager', 'IRkernel', 'devtools', 'tidyverse', 'flashClust', 'WGCNA', 'samr')
BiocManager::install(c('DEP', 'AnnotationDb', 'GO.db', 'preprocessCore', 'impute'))
IRkernel::installspec(user = FALSE) 