while true; do 
  echo -ne "$(date '+%Y-%m-%d\t%H:%M:%S')\t" >> ram_log.txt
  free -m | awk '/^Mem:/ {
      total=$2; available=$7
    } 
    END {
      used=total - available
      printf "%d\t%.2f\t%d\t%.2f\n", 
      used, used/total*100, available, available/total*100
    }' >> ram_log.txt
  sleep 60
done
