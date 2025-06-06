#!/bin/bash

IMAGE_NAME="pg_updater:1.0"
TODAY=$(date +%Y/%m/%d)

# Parse command line arguments
TEST_MODE=false
for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE=true
            shift
            ;;
    esac
done

# Parse volumes from docker-compose.yaml
echo "Parsing volume mappings from docker-compose.yaml..."
VOLUME_ARGS=()

# Extract the volume section and parse it
volumes=$(sed -n '/^[[:space:]]*volumes:/,/^[[:space:]]*[a-z]:/p' dockerfiles/docker-compose.yaml | grep "^[[:space:]]*-")

while IFS= read -r line; do
    # Extract source:destination from the volume line
    if [[ $line =~ [[:space:]]-[[:space:]]([^:]+):(.+) ]]; then
        SRC="${BASH_REMATCH[1]}"
        DEST="${BASH_REMATCH[2]}"
        VOLUME_ARGS+=("-v" "${SRC}:${DEST}")
    fi
done <<< "$volumes"

if [ ${#VOLUME_ARGS[@]} -eq 0 ]; then
    echo "Error: No volume mappings found in docker-compose.yaml"
    exit 1
fi

if [ "$TEST_MODE" = true ]; then
    echo "[$TODAY] Would run the following command:"
    echo "docker run --rm \\"
    for arg in "${VOLUME_ARGS[@]}"; do
        echo "    $arg \\"
    done
    echo "    ${IMAGE_NAME}"
else
    echo "[$TODAY] Running database updater container..."
    docker run --rm \
        "${VOLUME_ARGS[@]}" \
        "${IMAGE_NAME}"
fi
