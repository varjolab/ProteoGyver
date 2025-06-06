#!/bin/bash

LOG_DIR="${RESOURCE_LOG_DIR:-/tmp/resource_logs}"
INTERVAL="${RESOURCE_LOG_INTERVAL_SECONDS:-60}"
KEEP_DAYS="${RESOURCE_LOG_KEEP_DAYS:-7}"

mkdir -p "$LOG_DIR"

HEADER_BASE="Date\tTime\tUsed_MB\tUsed_%\tAvailable_MB\tAvailable_%"
HEADER_CPU="CPU_total"
# Get number of CPU cores and build header dynamically
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
for i in $(seq 0 $((NUM_CORES - 1))); do
  HEADER_CPU+="\tCPU$i"
done
HEADER="$HEADER_BASE\t$HEADER_CPU"

echo "[RESOURCE MONITOR] Logging to: $LOG_DIR"
echo "[RESOURCE MONITOR] Interval: $INTERVAL seconds | Keep logs: $KEEP_DAYS days | Cores: $NUM_CORES"

# Function to get CPU usage
get_cpu_stats() {
  cat /proc/stat | grep '^cpu' > /tmp/cpu_stat_1
  sleep 1
  cat /proc/stat | grep '^cpu' > /tmp/cpu_stat_2

  awk '
    NR==FNR && /^cpu/ {
      for (i=2; i<=NF; i++) prev[NR,i]=$i
      next
    }
    /^cpu/ {
      line = $1
      total=0; idle=0
      for (i=2; i<=NF; i++) {
        delta = $i - prev[FNR,i]
        total += delta
        if (i == 5) idle = delta  # idle time = 5th field
      }
      usage = (total - idle) / total * 100
      printf "%.1f\t", usage
    }
  ' /tmp/cpu_stat_1 /tmp/cpu_stat_2
}

while true; do
  TODAY=$(date +'%Y-%m-%d')
  LOG_FILE="$LOG_DIR/resources_${TODAY}.log"

  # Header if file doesn't exist or is missing the correct first line
  if [ ! -f "$LOG_FILE" ] || [ "$(head -n 1 "$LOG_FILE")" != "$(echo -e "$HEADER")" ]; then
    echo -e "$HEADER" > "$LOG_FILE"
  fi


  # Memory info
  MEM_LINE=$(free -m | awk '/^Mem:/ {
    total=$2; available=$7;
    used=total - available;
    printf "%d\t%.2f\t%d\t%.2f", used, used/total*100, available, available/total*100
  }')

  # CPU info
  CPU_LINE=$(get_cpu_stats)

  # Write log line
  echo -e "$(date '+%Y-%m-%d\t%H:%M:%S')\t$MEM_LINE\t$CPU_LINE" >> "$LOG_FILE"

  # Compress old logs
  find "$LOG_DIR" -type f -name "resources_*.log" ! -name "resources_${TODAY}.log" -exec gzip -f {} \;

  # Delete old compressed logs
  find "$LOG_DIR" -type f -name "resources_*.log.gz" -mtime +$KEEP_DAYS -delete

  sleep "$INTERVAL"
done
