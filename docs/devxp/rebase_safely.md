# How to Rebase Feature Branches Safely

This guide provides exact commands for safely rebasing feature branches to keep a clean commit history.

## Quick Reference (1-minute guide)

```bash
# 1. Fetch latest changes
git fetch origin

# 2. Switch to your feature branch
git checkout your-feature-branch

# 3. Rebase onto main (or target branch)
git rebase origin/main

# 4. If conflicts occur, resolve them and continue
git add .
git rebase --continue

# 5. Force push (only if you're the only one working on this branch)
git push --force-with-lease origin your-feature-branch
```

## Detailed Steps

### Before Starting
- Ensure your working directory is clean (`git status` should show no uncommitted changes)
- Make sure you're on the correct feature branch
- Verify the target branch (usually `main` or `develop`) is up to date

### Step 1: Fetch Latest Changes
```bash
git fetch origin
```
This downloads the latest commits from the remote repository without merging them.

### Step 2: Switch to Feature Branch
```bash
git checkout your-feature-branch
# or
git switch your-feature-branch
```

### Step 3: Start Interactive Rebase
```bash
# Rebase onto main branch
git rebase origin/main

# Or rebase onto develop branch
git rebase origin/develop

# For interactive rebase (allows squashing, editing commits)
git rebase -i origin/main
```

### Step 4: Handle Conflicts (if any)
If conflicts occur during rebase:

1. **Check status:**
   ```bash
   git status
   ```

2. **Resolve conflicts in your editor:**
   - Open conflicted files
   - Look for conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`)
   - Choose which changes to keep
   - Remove conflict markers

3. **Stage resolved files:**
   ```bash
   git add .
   # or
   git add path/to/resolved/file
   ```

4. **Continue rebase:**
   ```bash
   git rebase --continue
   ```

5. **Repeat until rebase is complete**

### Step 5: Abort if Needed
If you need to abort the rebase and return to the original state:
```bash
git rebase --abort
```

### Step 6: Push Changes
```bash
# Safe force push (recommended)
git push --force-with-lease origin your-feature-branch

# Regular force push (use with caution)
git push --force origin your-feature-branch
```

## Interactive Rebase Commands

When using `git rebase -i`, you can:
- `pick` - use the commit as-is
- `reword` - change the commit message
- `edit` - stop to amend the commit
- `squash` - combine with previous commit
- `drop` - remove the commit entirely

## Safety Tips

1. **Always use `--force-with-lease`** instead of `--force` to prevent overwriting others' work
2. **Test your changes** after rebasing to ensure everything still works
3. **Communicate with team** if others are working on the same branch
4. **Create a backup branch** before rebasing if unsure:
   ```bash
   git branch backup-branch-name
   ```

## Common Issues and Solutions

### "Your branch and 'origin/main' have diverged"
This means your local main is behind the remote. Fix with:
```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

### "Cannot rebase: You have unstaged changes"
Stash your changes first:
```bash
git stash
git rebase origin/main
git stash pop
```

### "This branch is up to date with 'origin/main'"
Your branch is already current. No rebase needed.

## Verification

After rebasing, verify everything is correct:
```bash
# Check commit history
git log --oneline -10

# Check that tests still pass
# (run your project's test suite)

# Check that the branch is ahead of main
git log origin/main..HEAD
```

## When NOT to Rebase

- **Never rebase shared/public branches** that others are working on
- **Don't rebase commits that have been pushed and pulled by others**
- **Avoid rebasing if you're unsure** - ask for help instead

## Emergency Recovery

If something goes wrong during rebase:
```bash
# Abort the rebase
git rebase --abort

# Or reset to a known good state
git reset --hard HEAD~5  # Go back 5 commits
git reset --hard origin/your-feature-branch  # Reset to remote state
```
