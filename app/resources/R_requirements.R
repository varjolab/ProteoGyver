install.packages(c('BiocManager', 'IRkernel', 'devtools', 'tidyverse', 'flashClust', 'samr'),Ncpus=16)
BiocManager::install(c('DEP'))
IRkernel::installspec(user = FALSE) 
install.packages(c('WGCNA'),Ncpus=16)