install.packages(c('BiocManager','devtools', 'proteomicsCV', 'vsn', 'IRkernel', 'devtools', 'tidyverse', 'flashClust', 'WGCNA', 'samr'), ncpus=12)
IRkernel::installspec(user = FALSE) 
library(devtools)
install_github("https://github.com/vdemichev/diann-rpackage")