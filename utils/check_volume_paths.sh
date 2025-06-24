#!/bin/bash

# Parse command line arguments
VERBOSE=false
CREATE_DIRS=false
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --create|-c)
            CREATE_DIRS=true
            shift
            ;;
    esac
done

echo "Checking volume source paths from docker-compose.yaml..."

# Extract the volume section and parse it
volumes=$(sed -n '/^[[:space:]]*volumes:/,/^[[:space:]]*[a-z]:/p' dockerfiles/docker-compose.yaml | grep "^[[:space:]]*-")

# Arrays to store results
declare -a MISSING_PATHS=()
declare -a EXISTING_PATHS=()
declare -a CREATED_PATHS=()

# Variable to store parameters.toml path
PARAMETERS_TOML=""

while IFS= read -r line; do
    # Extract source path from the volume line
    if [[ $line =~ [[:space:]]-[[:space:]]([^:]+):(.+) ]]; then
        SRC="${BASH_REMATCH[1]}"
        
        # Store parameters.toml path if found
        if [[ "$SRC" == *"parameters.toml" ]]; then
            PARAMETERS_TOML="$SRC"
        fi
        
        # If it's a .toml file, check its parent directory
        if [[ "$SRC" == *.toml ]]; then
            DIR=$(dirname "$SRC")
            if [ -d "$DIR" ]; then
                if [ "$VERBOSE" = true ]; then
                    EXISTING_PATHS+=("$DIR (for $SRC)")
                fi
            else
                if [ "$CREATE_DIRS" = true ]; then
                    mkdir -p "$DIR"
                    if [ $? -eq 0 ]; then
                        CREATED_PATHS+=("$DIR (for $SRC)")
                    else
                        MISSING_PATHS+=("$DIR (needed for $SRC) - Failed to create")
                    fi
                else
                    MISSING_PATHS+=("$DIR (needed for $SRC)")
                fi
            fi
        else
            # Regular directory check
            if [ -d "$SRC" ]; then
                if [ "$VERBOSE" = true ]; then
                    EXISTING_PATHS+=("$SRC")
                fi
            else
                if [ "$CREATE_DIRS" = true ]; then
                    mkdir -p "$SRC"
                    if [ $? -eq 0 ]; then
                        CREATED_PATHS+=("$SRC")
                    else
                        MISSING_PATHS+=("$SRC - Failed to create")
                    fi
                else
                    MISSING_PATHS+=("$SRC")
                fi
            fi
        fi
    fi
done <<< "$volumes"

# Print results
if [ ${#CREATED_PATHS[@]} -gt 0 ]; then
    echo "Created directories:"
    for path in "${CREATED_PATHS[@]}"; do
        echo "  + $path"
    done
fi

if [ ${#MISSING_PATHS[@]} -eq 0 ]; then
    echo "All required paths exist"
    if [ "$VERBOSE" = true ] && [ ${#EXISTING_PATHS[@]} -gt 0 ]; then
        echo "Existing paths:"
        for path in "${EXISTING_PATHS[@]}"; do
            echo "  ✓ $path"
        done
    fi
    echo "Copying over parameters.toml for the PG and updater containers"
    cp app/parameters.toml $PARAMETERS_TOML
    exit 0
else
    echo "Some required paths are missing:"
    for path in "${MISSING_PATHS[@]}"; do
        echo "  ✗ $path"
    done
    if [ "$VERBOSE" = true ] && [ ${#EXISTING_PATHS[@]} -gt 0 ]; then
        echo "Existing paths:"
        for path in "${EXISTING_PATHS[@]}"; do
            echo "  ✓ $path"
        done
    fi
    if [ "$CREATE_DIRS" = false ]; then
        echo "Tip: Use --create or -c to automatically create missing directories"
    fi
    exit 1
fi 