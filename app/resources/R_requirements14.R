readRenviron("/etc/default/locale")
LANG <- Sys.getenv("LANG")
if(nchar(LANG))
   Sys.setlocale("LC_ALL", LANG)
IRkernel::installspec(user = FALSE) 

