#!/bin/bash
# Skill Activation Hook - Shell Wrapper
# Runs the TypeScript skill matcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if tsx is available
if command -v npx &> /dev/null; then
    # Run the TypeScript hook
    npx tsx "$SCRIPT_DIR/skill-activation-prompt.ts"
else
    # Fallback: just pass through without skill matching
    exit 0
fi
