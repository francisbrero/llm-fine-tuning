#!/bin/bash
# Test Reminder Hook
# Reminds about testing when relevant files are modified

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/.edit-log.json"

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    exit 0
fi

# Get edited files
EDITS=$(jq -r '.edits[].file' "$LOG_FILE" 2>/dev/null | sort -u)

if [ -z "$EDITS" ]; then
    exit 0
fi

# Check for core model/training files
CORE_FILES=""
for FILE in $EDITS; do
    case "$FILE" in
        *run_evals.py*|*simple_test.py*|*fine_tune*.py*|*fine_tune*.ipynb*)
            CORE_FILES="$CORE_FILES $FILE"
            ;;
        *ccp_training*.jsonl*|*ccp_eval*.jsonl*)
            CORE_FILES="$CORE_FILES $FILE"
            ;;
    esac
done

if [ -n "$CORE_FILES" ]; then
    echo ""
    echo "=== TEST REMINDER ==="
    echo ""
    echo "Core files were modified:"
    for FILE in $CORE_FILES; do
        echo "  - $FILE"
    done
    echo ""
    echo "Recommended actions:"
    echo "  1. Run quick eval:  python run_evals.py --quick"
    echo "  2. Run full eval:   python run_evals.py --model finetuned"
    echo "  3. View history:    python eval_history.py"
    echo ""
    echo "See: .claude/skills/runbooks/run-evaluation.md"
fi

exit 0
