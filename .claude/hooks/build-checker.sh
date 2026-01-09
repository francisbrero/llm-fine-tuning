#!/bin/bash
# Build Checker Hook
# Runs on Stop to verify Python syntax and run tests if needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/.edit-log.json"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    exit 0
fi

# Get edited Python files
PYTHON_FILES=$(jq -r '.edits[] | select(.type == "python") | .file' "$LOG_FILE" 2>/dev/null | sort -u)

if [ -z "$PYTHON_FILES" ]; then
    exit 0
fi

echo ""
echo "=== BUILD CHECK ==="
echo ""

# Track if any checks failed
FAILED=0

# Check Python syntax
echo "Checking Python syntax..."
for FILE in $PYTHON_FILES; do
    if [ -f "$FILE" ]; then
        if ! python3 -m py_compile "$FILE" 2>&1; then
            echo "  [FAIL] Syntax error in $FILE"
            FAILED=1
        else
            echo "  [OK] $FILE"
        fi
    fi
done

# Check for training data files
DATA_FILES=$(jq -r '.edits[] | select(.type == "data") | .file' "$LOG_FILE" 2>/dev/null | sort -u)

if [ -n "$DATA_FILES" ]; then
    echo ""
    echo "Data files modified:"
    for FILE in $DATA_FILES; do
        echo "  - $FILE"
    done
    echo ""
    echo "Consider running: python run_evals.py --quick"
fi

# If all checks passed, clear the log
if [ $FAILED -eq 0 ]; then
    echo ""
    echo "[OK] All checks passed"
    # Clear the edit log
    echo '{"edits":[]}' > "$LOG_FILE"
else
    echo ""
    echo "[WARN] Some checks failed - please fix before committing"
fi

exit $FAILED
