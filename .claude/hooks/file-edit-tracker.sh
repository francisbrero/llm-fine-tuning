#!/bin/bash
# File Edit Tracker Hook
# Logs file edits for use by build-checker and test-reminder hooks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/.edit-log.json"

# Read tool info from stdin
read -r INPUT

# Parse tool name and file path
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)

# If we couldn't parse, try alternative format
if [ -z "$FILE_PATH" ]; then
    FILE_PATH=$(echo "$INPUT" | jq -r '.file_path // empty' 2>/dev/null)
fi

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Initialize log file if it doesn't exist
if [ ! -f "$LOG_FILE" ]; then
    echo '{"edits":[]}' > "$LOG_FILE"
fi

# Get timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Determine file type
FILE_EXT="${FILE_PATH##*.}"
case "$FILE_EXT" in
    py)
        FILE_TYPE="python"
        ;;
    ts|tsx|js|jsx)
        FILE_TYPE="typescript"
        ;;
    json|jsonl)
        FILE_TYPE="data"
        ;;
    md)
        FILE_TYPE="docs"
        ;;
    *)
        FILE_TYPE="other"
        ;;
esac

# Add edit to log
NEW_EDIT=$(cat <<EOF
{
  "timestamp": "$TIMESTAMP",
  "tool": "$TOOL_NAME",
  "file": "$FILE_PATH",
  "type": "$FILE_TYPE"
}
EOF
)

# Append to log (keep last 50 edits)
jq --argjson edit "$NEW_EDIT" '.edits = ([$edit] + .edits)[:50]' "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"

exit 0
