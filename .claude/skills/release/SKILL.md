---
name: release
description: Cut a new confpickem release -- picks the version bump from what merged since the last tag, drafts release notes in the house style, bumps pyproject.toml and __init__.py together, and publishes the GitHub release that triggers the PyPI upload.
argument-hint: "[patch | minor | major | X.Y.Z]"
disable-model-invocation: true
---

# Release confpickem

Publishing uploads to PyPI, which can't be undone (a version number can never
be reused). Get the user's explicit OK on the version and notes before step 5.
`gh` lives at `/opt/homebrew/bin/gh`.

## 1. Preflight

- On `main`, clean working tree, up to date with `origin/main`
  (`git fetch && git status -sb`). If not, stop and say what's off.
- `pytest --no-cov -q` passes. If anything fails, stop and show the failures.
- Current version: `version` in `pyproject.toml` and `__version__` in
  `src/confpickem/__init__.py` must match each other and the latest tag
  (`gh release list --limit 1`). If they disagree, stop and ask.

## 2. What's shipping

```bash
git log --first-parent --oneline v<CURRENT>..origin/main
gh pr list --state merged --base main --search "merged:>=<last release date>" --json number,title,body --limit 50
```

Read the PR titles and bodies, not just commit subjects. Ignore changes that
don't affect the installed package (`.claude/`, `docs/`, `scripts/`, tests only,
CI) when describing features, though they can be mentioned briefly.

If nothing package-facing merged, say so and ask whether to release anyway.

## 3. Version

Use the argument if given. Otherwise propose one (the project is pre-1.0):
- **minor** (0.X.0): new CLI modes, flags, optimizers, report features, or any
  behavior change a user would notice.
- **patch** (0.x.Y): bug fixes and internal changes only.
- **major**: only if the user asks.

## 4. Draft notes

Match the previous releases (`gh release view v<CURRENT>`):

```markdown
## What's New

### <Feature name>
- <user-facing bullet: what it does and how to use it (flag / mode)>

## Bug Fixes

- <what was wrong from the user's point of view, and what it affected>

## Breaking Changes

None - existing CLI flags and scripts continue to work.
```

Drop empty sections except Breaking Changes. Write for someone who uses the
CLI, not someone reading the diff.

Show the user the version and the full notes, and wait for approval or edits.

## 5. Bump, tag, publish

After approval:

1. Set the version in **both** `pyproject.toml` (`version = "X.Y.Z"`) and
   `src/confpickem/__init__.py` (`__version__ = 'X.Y.Z'`).
2. Commit on `main` with the message `Bump version to X.Y.Z` (house convention:
   the bump goes straight to main, no PR), then `git push origin main`.
3. Write the notes to a scratch file and:
   ```bash
   gh release create vX.Y.Z --target main --title "confpickem vX.Y.Z" --notes-file <file>
   ```
4. Publishing the release triggers `.github/workflows/publish.yml`. Watch it:
   ```bash
   gh run list --workflow publish.yml --limit 1
   gh run watch <run-id> --exit-status
   ```

## 6. Report

Give the release URL, the workflow result, and
`https://pypi.org/project/confpickem/X.Y.Z/`. If the publish job failed, show
the failing step's log (`gh run view <run-id> --log-failed`); the tag and
GitHub release already exist, so the fix is usually re-running the workflow,
not re-releasing.
