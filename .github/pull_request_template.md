# Pull Request Template

## Summary

Brief description of what this PR accomplishes and why it's needed.

## Strategic Impact

**Differentiator Tag**: [ ] [POA] [ ] [ASD] [ ] [RC] [ ] [ME] [ ] [IVL] [ ] [NSF] [ ] [FM]

**Strategic Value**: Brief explanation of how this PR advances our acquisition narrative

**Acquisition Narrative**: How this change demonstrates competitive positioning

**Measurable Outcomes**: Specific metrics, performance improvements, or capabilities gained

**Doctrine Alignment**: Reference to core technical doctrine (automation, algorithms, reliability, metrics, integration, security, formal methods)

## Scope

**Type**: [ ] Feature [ ] Bug Fix [ ] Performance [ ] Documentation [ ] Operations [ ] Quality Assurance

**Components Modified**:
- [ ] Backend (axiom_engine, logic, orchestrator, worker)
- [ ] Scripts (operations, maintenance, exports)
- [ ] Documentation (onboarding, runbooks, API reference)
- [ ] Configuration (CI, environment, deployment)
- [ ] Tests (unit tests, smoke tests, integration)

**Files Changed**: (List key files modified)
- `path/to/file1.py` - Brief description of changes
- `path/to/file2.md` - Brief description of changes

## Risk Assessment

**Risk Level**: [ ] Low [ ] Medium [ ] High

**Potential Impact**:
- [ ] Performance impact (specify expected change)
- [ ] Breaking changes (list affected APIs/interfaces)
- [ ] Database schema changes
- [ ] Configuration changes required
- [ ] Deployment considerations

**Rollback Plan**:
- [ ] Simple revert possible
- [ ] Requires data migration rollback
- [ ] Requires configuration rollback
- [ ] Other: (specify)

## Test Plan

### Unit Tests
```bash
# Commands run to verify changes
python -m pytest tests/test_specific_module.py -v
python -m backend.axiom_engine.derive --system pl --smoke-pl --seal
```

**Test Results**:
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Coverage maintained or improved
- [ ] Network-free test requirement met

### Integration Testing
- [ ] Smoke tests pass
- [ ] API endpoints functional
- [ ] Database operations successful
- [ ] Redis queue processing works

### Performance Testing (if applicable)
- [ ] Baseline performance maintained
- [ ] No memory leaks detected
- [ ] Response times within acceptable limits

## Conflict Watch

**Files Also Modified by Other PRs**:
- `file1.py` - PR #123 (author: @username) - Status: Open/Merged
- `file2.md` - PR #124 (author: @username) - Status: Open/Merged

**Coordination Notes**:
- [ ] Coordinated with other PR authors
- [ ] Merge order agreed upon
- [ ] No conflicts expected
- [ ] Conflicts resolved

## Checklist

### Code Quality
- [ ] Code follows project style guidelines
- [ ] ASCII-only content in docs/scripts
- [ ] No hardcoded secrets or credentials
- [ ] Error handling implemented
- [ ] Logging added where appropriate

### Documentation
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Inline code comments added (if complex logic)
- [ ] Migration notes included (if breaking changes)

### Security
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization considered
- [ ] Dependencies security reviewed

### Performance
- [ ] No significant performance regression
- [ ] Memory usage considered
- [ ] Database query optimization (if applicable)
- [ ] Caching strategy implemented (if applicable)

### Deployment
- [ ] Environment variables documented
- [ ] Database migrations included (if needed)
- [ ] Configuration changes documented
- [ ] Deployment instructions provided

## Additional Notes

Any additional context, screenshots, or information that reviewers should know.

### Screenshots (if UI changes)
<!-- Include before/after screenshots for UI changes -->

### Performance Metrics (if performance changes)
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Proofs/hour | 120 | 150 | +25% |
| Memory usage | 512MB | 480MB | -6.25% |

### Migration Notes (if applicable)
<!-- Include any migration steps or breaking changes -->

---

**Reviewer Notes**:
- This PR follows the contributing guidelines in `docs/CONTRIBUTING.md`
- Strategic differentiator tag is required ([POA], [ASD], [RC], [ME], [IVL], [NSF], [FM])
- All CI checks must pass before merge
- At least one approval required
