# Contributing to mlx-audio

Thanks for contributing to mlx-audio.

## Scope

We welcome:
- New model ports (TTS, STT, STS, SFX, VAD)
- Bug fixes and performance improvements
- Documentation and example improvements

For large API changes or new features, **open an issue first** to discuss the
approach before starting implementation. This avoids wasted effort on PRs that
won't be accepted.

## Reporting Bugs

Search [existing issues](https://github.com/Blaizzy/mlx-audio/issues) before
opening a new one. When reporting a bug, include:

- OS and Apple Silicon chip (e.g., macOS 15.3, M3 Pro)
- Python and MLX versions (`python --version`, `python -c "import mlx; print(mlx.__version__)"`)
- Full traceback
- Minimal reproducible snippet

## Security Vulnerabilities

Do **not** open a public issue for security vulnerabilities. Use
[GitHub Security Advisories](https://github.com/Blaizzy/mlx-audio/security/advisories/new)
to report privately.

## Development Setup

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[all,dev]"

# Install pre-commit hooks (required — CI uses pinned formatter versions)
pre-commit install
```

## Pull Requests

- Open pull requests against `Blaizzy/mlx-audio:main`.
- If you are contributing from a fork, make sure the base repository is
  `Blaizzy/mlx-audio` and the base branch is `main`.
- Keep pull requests focused. Include tests and documentation updates when
  behavior changes.
- Keep PRs atomic and touch the smallest possible amount of code. This helps
  reviewers evaluate and merge changes faster and with higher confidence.
- A checklist is pre-filled when you open a PR — fill it out before requesting review.

Run local checks before opening a PR:

```bash
# Formatting (always run via pre-commit — CI uses pinned 24.x version)
pre-commit run black --files <changed files>
pre-commit run isort --files <changed files>

# Core tests
pytest -s mlx_audio/tests/
```

## Keeping the repository clean

Do not commit personal or temporary files. Before opening a PR, make sure your
diff does not include:

- Planning docs, notes, or spec files (`docs/plans/`, `TODO.md`, etc.)
- Test scripts, scratch notebooks, or one-off debug files
- Local model weights, audio samples, or large binaries
- Changes to `.gitignore` that only cover your personal setup

**Use a global gitignore for personal patterns** so you never have to touch the
project's `.gitignore`:

```bash
# Create (or append to) your global gitignore
echo "*.local.*" >> ~/.gitignore_global
echo ".env.local" >> ~/.gitignore_global

# Register it with git (one-time setup)
git config --global core.excludesFile ~/.gitignore_global
```

The `*.local.*` pattern (e.g., `TODO.local.md`, `config.local.json`) is a
useful convention for files that should always stay local.

## Adding a New Model

See [ADDING_A_MODEL.md](ADDING_A_MODEL.md) for the full guide — directory
layout, required class interfaces, `generate()` kwargs, weight conversion,
codec integration, tests, and PR checklist.

## Good First Issues

Issues labeled [`good first issue`](https://github.com/Blaizzy/mlx-audio/contribute)
are a good starting point if you are new to the codebase.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold it.

## Commit Signing and Account Security

To improve commit provenance and reduce supply chain risk, please sign commits
submitted to this repository. This is a one-time setup on your machine.

- Any GitHub-supported signing method is fine: GPG, SSH, or S/MIME.
- Enable GitHub vigilant mode so commits and tags always show a verification
  status.
- Enable two-factor authentication on your GitHub account. Passkeys are
  preferred when available.

## References

- [About commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- [Displaying verification statuses for all of your commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits)
- [Enable vigilant mode](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits#enabling-vigilant-mode)
- [GPG setup walkthrough](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key)
