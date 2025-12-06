# Contributing to MathLedger

Guidelines for contributing to the MathLedger automated theorem proving system.

## Development Workflow TL;DR

Quick reference for contributing to MathLedger:

- **Branch**: Use prefixed names (`feature/`, `perf/`, `ops/`, `qa/`, `devxp/`, `docs/`) - no direct pushes to main
- **Pre-commit**: Run `git add specific-files` (never `git add .`) and ensure ASCII-only content in docs/scripts
- **PR Template**: Fill out all sections (Summary, Scope, Risk, Test Plan, Conflict Watch) completely
- **ASCII-Only**: All documentation and scripts must contain only ASCII characters (no smart quotes, Unicode)
- **CI Expectations**: All automated checks must pass before merge (hygiene, tests, coverage floor)

## Branch Naming Convention

Use descriptive, hyphenated branch names with the following prefixes:

- `feature/` - New features and functionality
- `perf/` - Performance improvements and optimizations
- `ops/` - Operational changes (CI, deployment, monitoring)
- `qa/` - Quality assurance, testing improvements
- `devxp/` - Developer experience improvements
- `docs/` - Documentation updates and additions

Examples:
```
feature/fol-guided-derivation
perf/redis-connection-pooling
ops/nightly-pipeline-alerts
qa/smoke-test-coverage
devxp/bootstrap-script-improvements
docs/api-reference-update
```

## Development Workflow

### 1. No Direct Pushes to Main

All changes must go through pull requests. Direct pushes to `main` are prohibited.

```bash
# Correct workflow
git checkout main
git pull --ff-only
git checkout -b feature/your-feature-name
# ... make changes ...
git add specific-files
git commit -m "feat: descriptive commit message"
git push -u origin feature/your-feature-name
# Create PR via GitHub UI or gh cli
```

### 2. Pull Request Requirements

Before creating a PR, ensure:

- **CI Must Pass**: All automated checks must pass before merge
- **ASCII-Only Policy**: All documentation and scripts must contain only ASCII characters
- **No Artifacts**: Do not commit `artifacts/`, `dist/`, or build directories
- **File Size Limits**: No files larger than 2MB
- **Network-Free Tests**: Unit tests must not require external network access

### 3. Commit Message Format

Use conventional commit format:

```
type(scope): description

feat(derive): add guided derivation mode
fix(api): resolve authentication timeout
docs(onboarding): update 30-minute guide
perf(worker): optimize Redis job processing
ops(ci): add coverage floor enforcement
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `ops`, `ci`

## Code Quality Standards

### ASCII-Only Content

All documentation and scripts must use only ASCII characters:

- No smart quotes (" " instead of " ")
- No em dashes (-- instead of --)
- No Unicode symbols or accented characters
- Use standard ASCII punctuation and formatting

### Testing Requirements

- **Unit Tests**: All new functionality must include unit tests
- **Network-Free**: Tests must use mocks for external dependencies
- **Coverage**: Do not decrease test coverage by more than 1%
- **Smoke Tests**: Verify core functionality works end-to-end

### Documentation Standards

- **Truth-of-Main**: Document current functionality, not future plans
- **Migration Notes**: Use callout boxes for pending features
- **Mermaid Diagrams**: Use text-based diagrams, no binary images
- **Runnable Examples**: Include working command examples

## PR Gating Process

### Automated Checks

1. **Hygiene Enforcement**
   - ASCII compliance for docs/scripts
   - File size limits (<2MB)
   - Artifact directory blocking
   - Pre-commit hook validation

2. **Code Quality**
   - Black formatting
   - isort import sorting
   - flake8 linting
   - Type checking (where applicable)

3. **Testing**
   - Unit test execution
   - Coverage floor validation
   - Network isolation verification

### Manual Review

- **Architecture Alignment**: Changes align with system design
- **Performance Impact**: No significant performance regressions
- **Security Review**: No secrets or sensitive data exposed
- **Documentation**: Changes are properly documented

## Development Environment

### Quick Setup

```bash
# Clone and bootstrap
git clone https://github.com/helpfuldolphin/mathledger.git
cd mathledger
./scripts/bootstrap.sh  # Linux/macOS
# or: scripts/bootstrap.ps1  # Windows

# Verify setup
python -m backend.axiom_engine.derive --system pl --smoke-pl --seal
```

### Starter Kit

Essential commands for daily development workflow:

#### How to Start a Branch

**PowerShell:**
```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Create feature branch
git checkout main
git pull --ff-only
git checkout -b feature/your-feature-name
```

**Ubuntu:**
```bash
# Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Create feature branch
git checkout main
git pull --ff-only
git checkout -b feature/your-feature-name
```

#### How to Run Checks Locally

**PowerShell:**
```powershell
# Network-free unit tests
$env:NO_NETWORK="true"
pytest -q -k "not integration and not derive_cli"

# Coverage check (70% minimum)
coverage run -m unittest
coverage report --fail-under=70

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

**Ubuntu:**
```bash
# Network-free unit tests
export NO_NETWORK=true
pytest -q -k "not integration and not derive_cli"

# Coverage check (70% minimum)
coverage run -m unittest
coverage report --fail-under=70

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

#### PR Template Usage

When creating a PR, fill out all sections in `.github/pull_request_template.md`:
- **Summary**: Brief description of changes and rationale
- **Scope**: Components modified, risk level, rollback plan
- **Test Plan**: Unit tests, integration tests, performance validation
- **Conflict Watch**: Coordinate with other PRs touching same files

See the template for complete checklists and requirements.

## Strategic PR Differentiators

Every PR must declare its strategic differentiator to ensure each contribution advances our acquisition narrative. All merged PRs must carry one of the following differentiator tags and explicit doctrine references.

### Differentiator Tags

#### [POA] - Proof of Automation
Demonstrates autonomous theorem proving capabilities that reduce human intervention.

**Examples:**
- `feat(derive): implement guided derivation with 95% success rate [POA]`
- `perf(worker): optimize proof queue processing for 3x throughput [POA]`
- `fix(engine): resolve timeout issues in complex FOL derivations [POA]`

**Doctrine Reference:** Automated reasoning systems that scale beyond human capacity.

#### [ASD] - Algorithmic Superiority Demonstration
Shows novel algorithmic approaches that outperform existing methods.

**Examples:**
- `feat(canon): implement advanced normalization with 40% speed improvement [ASD]`
- `perf(redis): introduce connection pooling reducing latency by 60% [ASD]`
- `feat(fol): add congruence closure algorithm for equality reasoning [ASD]`

**Doctrine Reference:** Proprietary algorithms that create competitive moats.

#### [RC] - Reliability & Correctness
Enhances system reliability, correctness verification, or error handling.

**Examples:**
- `test(integration): add comprehensive smoke test coverage [RC]`
- `fix(api): implement proper error handling for malformed requests [RC]`
- `feat(validation): add Merkle tree verification for proof integrity [RC]`

**Doctrine Reference:** Mission-critical reliability for production deployment.

#### [ME] - Metrics & Evidence
Provides measurable evidence of system performance and capabilities.

**Examples:**
- `feat(metrics): implement v1 schema with backward compatibility [ME]`
- `ops(monitoring): add performance dashboards for proof throughput [ME]`
- `docs(evidence): document 85% success rate benchmarks [ME]`

**Doctrine Reference:** Quantifiable performance metrics for stakeholder confidence.

#### [IVL] - Integration & Validation Layer
Improves system integration, API design, or validation frameworks.

**Examples:**
- `feat(api): add RESTful endpoints for proof submission [IVL]`
- `ops(ci): implement automated testing pipeline [IVL]`
- `feat(export): add JSON export format for external systems [IVL]`

**Doctrine Reference:** Seamless integration with existing enterprise systems.

#### [NSF] - Network Security & Forensics
Enhances security, audit trails, or forensic capabilities.

**Examples:**
- `feat(auth): implement JWT-based authentication [NSF]`
- `ops(audit): add comprehensive logging for proof operations [NSF]`
- `fix(security): resolve SQL injection vulnerabilities [NSF]`

**Doctrine Reference:** Enterprise-grade security for sensitive operations.

#### [FM] - Formal Methods
Advances formal verification, mathematical rigor, or theoretical foundations.

**Examples:**
- `feat(lean): integrate Lean 4 formal verification backend [FM]`
- `docs(theory): document propositional logic completeness proofs [FM]`
- `feat(verify): add automated proof checking against Lean theorems [FM]`

**Doctrine Reference:** Mathematical rigor that ensures theoretical soundness.

### Strategic PR Requirements

#### Mandatory Elements

1. **Differentiator Tag**: Include exactly one tag ([POA], [ASD], [RC], [ME], [IVL], [NSF], [FM]) in commit message and PR title
2. **Doctrine Reference**: Explicitly connect changes to acquisition narrative
3. **Measurable Impact**: Quantify improvements where applicable (performance, coverage, success rates)
4. **Strategic Context**: Explain how this PR advances competitive positioning

#### PR Title Format

```
[TAG] type(scope): description with quantified impact

Examples:
[POA] feat(derive): implement guided derivation with 95% success rate
[ASD] perf(canon): optimize normalization reducing processing time by 40%
[RC] test(integration): achieve 90% test coverage for core derivation engine
[ME] feat(metrics): add v1 schema supporting 10K+ proofs/hour monitoring [hash:e8f4a2c1]
[IVL] feat(api): add RESTful endpoints enabling external system integration
[NSF] ops(audit): implement comprehensive logging for SOC2 compliance
[FM] feat(lean): integrate formal verification ensuring mathematical soundness
```

#### Strategic Impact Statement

Each PR description must include a "Strategic Impact" section:

```markdown
## Strategic Impact

**Differentiator**: [TAG] - Brief explanation of strategic value
**Acquisition Narrative**: How this advances our competitive positioning
**Measurable Outcomes**: Specific metrics or capabilities gained
**Doctrine Alignment**: Reference to core technical doctrine
```

### Acquisition Narrative Framework

Every PR contributes to our acquisition story by demonstrating:

1. **Technical Superiority**: Novel algorithms and automated reasoning capabilities
2. **Production Readiness**: Reliability, security, and enterprise integration
3. **Measurable Performance**: Quantified improvements and benchmarks
4. **Formal Rigor**: Mathematical soundness and verification capabilities
5. **Competitive Moats**: Proprietary techniques and algorithmic advantages

### Enforcement

- **100% Compliance Required**: All merged PRs must carry differentiator tags
- **Review Gate**: PRs without strategic differentiators will be rejected
- **Acquisition Readiness**: Each sprint must demonstrate progress across all differentiator categories
- **Narrative Coherence**: PRs must explicitly connect to broader acquisition story

> **Announcement**: See GitHub issue template `.github/ISSUE_TEMPLATE/strategic-differentiator-announcement.md` for the official announcement of this requirement.

### Dependencies

Minimal required dependencies:
- Python 3.8+
- fastapi, uvicorn, redis, psycopg, pydantic

Optional for full development:
- PostgreSQL (for database operations)
- Redis (for worker queue)
- Lean 4 (for formal verification)

### IDE Configuration

Recommended VS Code extensions:
- Python
- Black Formatter
- isort
- Mermaid Preview

## Conflict Resolution

### Merge Conflicts

When encountering merge conflicts:

1. **Fetch Latest**: `git fetch origin main`
2. **Rebase**: `git rebase origin/main` (preferred over merge)
3. **Resolve Conflicts**: Edit files to resolve conflicts
4. **Test**: Run full test suite after resolution
5. **Force Push**: `git push --force-with-lease origin your-branch`

### Shared File Changes

If your PR touches files also modified by other PRs:

1. **List Conflicts**: Document in PR description which files overlap
2. **Coordinate**: Communicate with other PR authors
3. **Sequential Merge**: Agree on merge order to minimize conflicts
4. **Rebase After**: Rebase your branch after conflicting PRs merge

## Release Process

### Version Tagging

- Use semantic versioning: `v1.2.3`
- Tag format: `mvdppass-YYYYMMDD-HHMM` for milestone releases
- Include release notes with performance metrics

### Deployment

- **Staging**: All PRs deploy to staging environment
- **Production**: Only tagged releases deploy to production
- **Rollback**: Maintain rollback capability for all deployments

## Support and Communication

### Getting Help

- **Documentation**: Start with `docs/onboarding/first_30_minutes.md`
- **API Reference**: See `docs/API_REFERENCE.md`
- **Runbooks**: Check `docs/runbooks/` for operational procedures
- **Issues**: Create GitHub issues for bugs and feature requests

### Code Review

- **Timely Reviews**: Respond to review requests within 24 hours
- **Constructive Feedback**: Focus on code quality and maintainability
- **Knowledge Sharing**: Explain reasoning behind suggestions
- **Approval Required**: At least one approval required before merge

## Performance Standards

### Acceptance Criteria

- **Proof Success Rate**: >=85% for production runs
- **Performance Baseline**: >=40 proofs/hour baseline [hash:e8f4a2c1], >=120 proofs/hour guided [hash:e8f4a2c1]
- **Timeout Limits**: Individual proof attempts <=5000ms
- **Block Validation**: Valid Merkle root and sequential block numbers

### Monitoring

- **Metrics Collection**: All changes must maintain metrics collection
- **Performance Regression**: No >10% performance degradation
- **Resource Usage**: Monitor CPU, memory, and disk usage
- **Error Rates**: Track and minimize error rates

## Security Guidelines

### Secrets Management

- **No Hardcoded Secrets**: Use environment variables
- **Key Rotation**: Regular rotation of API keys and tokens
- **Access Control**: Principle of least privilege
- **Audit Trail**: Log all security-relevant operations

### Data Protection

- **Input Validation**: Validate all external inputs
- **SQL Injection**: Use parameterized queries
- **XSS Prevention**: Sanitize web interface inputs
- **HTTPS Only**: All external communications over HTTPS

---

For questions about contributing, see the onboarding documentation or create a GitHub issue.
