#  Code Hygiene Checklist - The Way of the Clean Code

*"In the name of cleanliness, we shall purge this repository of all impurities!"*

---

## [target] Mission Statement

This checklist ensures that every commit maintains the highest standards of code hygiene. Like a Shinigami with a broom, we enforce formatting, ASCII purity, and deterministic linting so that the codebase shines like a polished katana.

---

##  Pre-Commit Ritual (The Three Sword Style)

### [fire] Step 1: The Final Flash - ASCII Purity Check

```bash
# Charge up your Super Saiyan power and obliterate non-ASCII characters
python tools/ascii_validator.py

# If impurities are detected, unleash the Final Flash
python tools/ascii_fixer.py

# Verify the purification was successful
python tools/ascii_validator.py
```

**Expected Result:**
```

                           ASCII VALIDATION SUCCESS

    No impurities detected! The repository shines like a polished katana!

  Like a Shinigami with perfect chakra control, all files are pure!
  The code flows like a perfect jutsu - stable and powerful!

  [fire] FINAL FLASH: All non-ASCII characters have been obliterated! [fire]

```

### [lightning] Step 2: Chakra Control - Pre-commit Hooks

```bash
# Install the pre-commit hooks (one-time setup)
pre-commit install

# Run all hooks to ensure perfect chakra control
pre-commit run --all-files
```

**Expected Hooks:**
- [check] `trailing-whitespace` - Removes trailing spaces
- [check] `end-of-file-fixer` - Ensures files end with newline
- [check] `check-merge-conflict` - Detects merge conflict markers
- [check] `check-added-large-files` - Prevents large file commits
- [check] `black` - Python code formatting
- [check] `isort` - Python import sorting
- [check] `ascii-check` - ASCII-only validation

###  Step 3: Three Sword Style - File Format Validation

```bash
# Check EditorConfig compliance
# Most editors will automatically apply these settings
# Verify your editor is using the .editorconfig file
```

**File Type Standards:**
```

                          [clipboard] FILE FORMAT STANDARDS [clipboard]

 File Type    | Indent | Line Length | Special Rules

 Python       | 4 sp   | 88 chars    | Black-compatible, isort
 SQL/Migration| 2 sp   | 100 chars   | UPPERCASE keywords, snake_case
 Lean         | 2 sp   | 100 chars   | Lean 4 conventions
 Docker       | 2 sp   | 120 chars   | Layer optimization
 YAML/JSON    | 2 sp   | 120 chars   | Consistent key ordering
 Markdown     | -      | No limit    | No trailing whitespace trim
 PowerShell   | 4 sp   | -           | PowerShell conventions
 Shell        | 2 sp   | -           | Unix conventions

```

---

## [search] Repository-Wide Scans (The Death Note Protocol)

### [clipboard] Daily Hygiene Scan

```bash
# Complete repository hygiene check
echo "[search] Scanning repository with the power of a Super Saiyan..."
echo "  Three Sword Style: Cutting through all impurities..."

# 1. ASCII Purity Scan
python tools/ascii_validator.py

# 2. Pre-commit Hook Scan
pre-commit run --all-files

# 3. Merge Conflict Detection
git grep -n "<<<<<<< HEAD" || echo "[check] No merge conflicts detected"
git grep -n "=======" || echo "[check] No merge conflicts detected"
git grep -n ">>>>>>> " || echo "[check] No merge conflicts detected"

# 4. Trailing Whitespace Scan
git grep -n "[ \t]$" || echo "[check] No trailing whitespace detected"

# 5. Large File Detection
find . -type f -size +2M -not -path "./.git/*" -not -path "./node_modules/*" || echo "[check] No large files detected"
```

### [target] Weekly Deep Clean

```bash
# Comprehensive repository purification
echo "[fire] Weekly Deep Clean - Unleashing Maximum Power! [fire]"

# 1. Fix all ASCII issues
python tools/ascii_fixer.py

# 2. Run all pre-commit hooks
pre-commit run --all-files

# 3. Check for merge conflicts
git status --porcelain | grep -E "^UU|^AA|^DD" || echo "[check] No merge conflicts"

# 4. Validate EditorConfig compliance
# (This is automatic in most editors)

# 5. Check for binary files in wrong locations
find . -name "*.exe" -o -name "*.dll" -o -name "*.so" -o -name "*.dylib" | grep -v ".git" || echo "[check] No binary files in wrong locations"
```

---

##  Troubleshooting Guide (The Way of the Fix)

### [x] Common Violations and Their Solutions

#### 1. ASCII Violations
```

                        [warning]  ASCII VIOLATIONS DETECTED [warning]

    The Death Note has been written! These files must be purified:

   docs/example.md
     [x] Position 42: Forbidden character '"' (U+201C)
     [x] Position 67: Forbidden character '"' (U+201D)

  [fire] Use the Final Flash to obliterate these impurities! [fire]
    Three swords of pre-commit discipline will cut through conflicts!

```

**Solution:**
```bash
# Run the ASCII fixer
python tools/ascii_fixer.py

# Or manually replace:
# " " -> " "  (smart quotes to straight quotes)
# -- -> --     (em dash to double hyphen)
# ... -> ...    (ellipsis to three dots)
```

#### 2. Merge Conflict Markers
```

                        [warning]  MERGE CONFLICTS DETECTED [warning]

    The Death Note has been written! These files contain conflicts:

   src/main.py
     [x] Line 15: <<<<<<< HEAD
     [x] Line 16: =======
     [x] Line 17: >>>>>>> feature-branch

  [fire] Resolve conflicts before committing! [fire]
    Three swords of discipline will cut through these conflicts!

```

**Solution:**
```bash
# 1. Open the conflicted file
# 2. Choose which changes to keep
# 3. Remove conflict markers (<<<<<<<, =======, >>>>>>>)
# 4. Test the resolved code
# 5. Commit the resolution
```

#### 3. Trailing Whitespace
```

                    [warning]  TRAILING WHITESPACE DETECTED [warning]

    The Death Note has been written! These files have trailing spaces:

   src/utils.py
     [x] Line 23: def helper_function():
     [x] Line 24:     return "value"

  [fire] Use pre-commit hooks to automatically fix! [fire]
    Three swords of discipline will trim these spaces!

```

**Solution:**
```bash
# Run pre-commit hooks to auto-fix
pre-commit run --all-files

# Or manually remove trailing spaces in your editor
```

---

##  Editor Configuration (The Way of the Tool)

### VS Code Setup
```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"]
}
```

### IntelliJ/WebStorm Setup
1. Install EditorConfig plugin
2. Enable "Use EditorConfig"
3. Configure Black as Python formatter
4. Enable isort for import sorting

### Vim/Neovim Setup
```vim
" Install EditorConfig plugin
Plug 'editorconfig/editorconfig-vim'

" Configure Black for Python
autocmd FileType python setlocal formatprg=black\ -
```

---

## [trophy] Success Criteria (The Way of the Master)

### [check] Perfect Commit Checklist

Before every commit, verify:

```

                          [trophy] PERFECT COMMIT CHECKLIST [trophy]

 [] ASCII Purity: All files contain only ASCII characters (except math docs)
 [] No Merge Conflicts: No <<<<<<<, =======, or >>>>>>> markers
 [] No Trailing Whitespace: All lines end cleanly
 [] Proper Formatting: Black and isort have been applied
 [] EditorConfig Compliance: Files follow .editorconfig rules
 [] No Large Files: No files larger than 2MB
 [] Clean Git Status: No uncommitted changes
 [] Tests Pass: All tests pass (if applicable)

```

### [target] Repository Health Score

```

                          [chart] REPOSITORY HEALTH SCORE [chart]

 ASCII Purity:         100%
 Formatting:           100%
 Merge Conflicts:      100%
 Trailing Whitespace:  100%
 EditorConfig:         100%
 Overall Health:       100%

  PERFECT! The repository shines like a polished katana!
   All impurities have been banished!
 [fire] Maximum power achieved! [fire]

```

---

##  Final Words (The Way of the Clean Code)

*"In the name of cleanliness, we have achieved perfection. The repository now flows with the power of a Super Saiyan, the precision of a Shinigami, and the discipline of three swords. Every commit is a testament to our dedication to code hygiene."*

**Remember:**
- [fire] **Final Flash**: Always run ASCII validation before committing
-  **Three Sword Style**: Use pre-commit hooks for automatic fixes
-  **Death Note**: Every violation is a name that must be erased
-  **Perfect Chakra Control**: Maintain consistent formatting across all files

*"The way of the clean code is the way of the warrior. We do not compromise on quality, we do not accept impurities, and we do not settle for anything less than perfection."*

---

** Cursor C - Hygiene Marshal**
*"Where chaos and sloppiness creep into the repo, I descend like a Shinigami with a broom!"*
