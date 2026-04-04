################################################################################
# Config Script
# Adverse Reactions to the Use of Large Language Models in Social Interactions
# F. Dvorak, R. Stumpf, S. Fehrler & U. Fischbacher
# PNAS Nexus
################################################################################
# This lists the libraries that are to be installed.
global.libraries <- c("dplyr", "reshape2", "stringr", "readr", "tidyverse", 
                      "readxl", "ds4psy", "kableExtra", "unikn", "DescTools", 
                      "hunspell", "openxlsx", "ggsignif", "rstatix", "jtools",
                      "stargazer", "gtools", "ggpubr", "gridExtra", "cowplot",
                      "glmnet", "ggplot2")


# Function to install libraries
pkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE)
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
  return("OK")
}
## Add any libraries to this line, and uncomment it.
results <- sapply(as.list(global.libraries), pkgTest)
print(sessionInfo())