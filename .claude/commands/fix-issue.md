# Fix Issue Command

Fetch a GitHub issue and implement it with proper workflow.

## Usage
```
/fix-issue <issue-number>
```

## Workflow

### Step 1: Fetch Issue Details
```bash
gh issue view $ARGUMENTS --json title,body,labels,assignees,milestone
```

### Step 2: Create Branch
Based on issue labels, create an appropriate branch:
- `bug` label → `bugfix/issue-{number}-{slug}`
- `feature` label → `feature/issue-{number}-{slug}`
- `enhancement` label → `feature/issue-{number}-{slug}`
- `docs` label → `docs/issue-{number}-{slug}`
- Default → `fix/issue-{number}-{slug}`

```bash
git checkout -b <branch-name>
```

### Step 3: Create Dev Docs
Create a task folder for context persistence:
```
dev/active/issue-{number}/
├── plan.md      # Implementation plan
├── context.md   # Current state and key files
└── tasks.md     # Checklist with status
```

### Step 4: Plan Implementation
Analyze the issue and create a phased implementation plan:

1. **Research Phase**: Identify affected files, understand current behavior
2. **Design Phase**: Outline changes needed, identify edge cases
3. **Implementation Phase**: Make changes in logical order
4. **Testing Phase**: Write/update tests, run test suite
5. **Documentation Phase**: Update relevant docs if needed

**IMPORTANT**: Present the plan to the user and WAIT for approval before implementing.

### Step 5: Implement
After user approval:
1. Make changes according to plan
2. Update `dev/active/issue-{number}/context.md` with progress
3. Check off tasks in `tasks.md` as completed

### Step 6: Verify
Before finalizing:
```bash
# Run type checking
python -m py_compile <modified-files>

# Run tests
python -m pytest tests/ -v

# Run linting if configured
ruff check . --fix
```

### Step 7: Create PR
```bash
git add -A
git commit -m "$(cat <<'EOF'
Fix #{issue-number}: {title}

{summary of changes}

Closes #{issue-number}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

gh pr create --title "Fix #{issue-number}: {title}" --body "$(cat <<'EOF'
## Summary
{bullet points of changes}

## Test Plan
- [ ] {test items}

Closes #{issue-number}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### Step 8: Cleanup
Move dev docs to completed:
```bash
mv dev/active/issue-{number} dev/completed/
```

---

## Templates

### plan.md
```markdown
# Issue #{number}: {title}

## Overview
{Brief description of the issue}

## Affected Files
- `path/to/file1.py` - {why}
- `path/to/file2.py` - {why}

## Implementation Phases

### Phase 1: {name}
- [ ] Task 1
- [ ] Task 2

### Phase 2: {name}
- [ ] Task 1
- [ ] Task 2

## Edge Cases
- {edge case 1}
- {edge case 2}

## Testing Strategy
- {test approach}
```

### context.md
```markdown
# Context: Issue #{number}

## Current State
{description of current progress}

## Key Files
- `file1.py:123` - {what's here}
- `file2.py:45` - {what's here}

## Recent Changes
- {change 1}
- {change 2}

## Next Steps
1. {next step}
2. {next step}

## Blockers
- {blocker if any}
```

### tasks.md
```markdown
# Tasks: Issue #{number}

## Status: In Progress

### Research
- [x] Reviewed issue requirements
- [x] Identified affected files
- [ ] Understood current behavior

### Implementation
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Testing
- [ ] Updated/added tests
- [ ] All tests passing
- [ ] Manual verification

### Finalization
- [ ] Code review ready
- [ ] PR created
- [ ] Dev docs archived
```
