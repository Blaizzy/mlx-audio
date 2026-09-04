# Releasing MLX-Audio

MLX-Audio releases are prepared by the `Release` GitHub Actions workflow. The
workflow owns the version change, release pull request, tag, GitHub Release,
package build, and PyPI publication.

Do not manually change `mlx_audio/version.py`, create a release tag, or upload a
distribution to PyPI.

## Release model

Ordinary pull requests accumulate on `main` without starting a release. A
maintainer explicitly starts a release cycle with a `Release-As` commit footer
or a manual workflow run. Release Please then opens one release pull request.

While that pull request is open, Release Please keeps its version and changelog
synchronized with changes that continue to land on `main`. Publication happens
only when a maintainer merges the release pull request.

| Action | Result |
| --- | --- |
| Merge an ordinary pull request | Accumulates a change without starting a release |
| Land a valid `Release-As` footer | Creates or updates the release pull request |
| Run the workflow without `release_as` | Calculates a version from accumulated Conventional Commits |
| Run the workflow with `release_as` | Validates and requests that exact version |
| Merge the release pull request | Tags, builds, verifies, and publishes the package |

## Contributor responsibilities

Contributors should use a Conventional Commit pull-request title such as
`fix(tts): ...`, `feat(stt): ...`, or `docs: ...`. These titles determine the
generated changelog and the automatically calculated version.

The `Release-As` footer is a maintainer control. Contributors should not add it
unless a maintainer specifically requests it.

## Start a release from a commit

Land a commit on `main` whose body contains an exact version:

```text
chore: prepare release

Release-As: NEXT_VERSION
```

The footer must remain in the final commit message that reaches `main`. The
workflow reads the current version from `.release-please-manifest.json`; no
version is hardcoded in the workflow.

For example, if the current manifest version is `0.5.0`, valid release lines
include:

- Patch: `0.5.1`
- Minor: `0.6.0`
- Major: `1.0.0`
- Minor release candidate: `0.6.0-rc1`

The guard rejects malformed versions, downgrades, duplicate versions, skipped
patch or minor lines, and release-candidate regressions. From `0.3.0`, for
example, `0.7.0` is rejected because the next minor line is `0.4.0`.

Supported versions use `MAJOR.MINOR.PATCH` or `MAJOR.MINOR.PATCH-rcN` syntax.
Python package metadata normalizes `0.6.0-rc1` to `0.6.0rc1`.

## Start a release manually

1. Open the repository's **Actions** tab.
2. Select the **Release** workflow.
3. Choose **Run workflow** on `main`.
4. Leave `release_as` blank or enter an exact supported version.
5. Run the workflow and review the resulting release pull request.

Leaving `release_as` blank derives the next version from all accumulated
Conventional Commits. Entering a version applies the same transition guard as a
`Release-As` footer. A manual run never publishes directly.

## Review the release pull request

Release Please updates:

- `mlx_audio/version.py`
- `.release-please-manifest.json`
- `CHANGELOG.md`

Before merging, confirm:

- The version file, manifest, release title, and changelog agree.
- The changelog covers the intended TTS, STT, STS, LID, codec, and VAD changes
  without unrelated entries.
- Release candidates use `-rcN` in the source and normalize correctly in Python
  package metadata.
- Every commit is signed and the PR-title check passes.
- The documentation build and pre-commit style job pass.
- Core, parity, and modular `stt`, `tts`, and `sts` installation checks pass.
- The TTS, STT, STS, LID, codec, and VAD test matrices pass.
- The package version is the one maintainers intend to publish.

Do not manually edit the bot-managed release pull request. Correct the source
commit or run the workflow again so Release Please updates it consistently.

## Publication sequence

Merging the release pull request causes Release Please to create the `vVERSION`
tag and GitHub Release. The workflow then:

1. Checks out the release tag.
2. Builds the wheel and source distribution once.
3. Runs Twine validation and installs the wheel to verify its package metadata.
4. Attaches the distributions to the GitHub Release.
5. Publishes those same workflow artifacts to PyPI using Trusted Publishing.

The PyPI environment remains the final publication boundary; ordinary pull
requests and unmerged release pull requests cannot publish packages.

## Recover a failed release

- **Version guard failure:** read the workflow error and submit a valid next
  patch, minor, major, or release-candidate transition. Do not edit the
  manifest manually.
- **Release pull-request CI failure:** fix the underlying change on `main` and
  let Release Please synchronize the open release pull request.
- **Build or GitHub upload failure:** rerun the failed build job for the existing
  tag. Do not create a replacement tag or version.
- **PyPI failure:** first inspect PyPI to determine whether any distribution was
  accepted. Rerun publishing only when doing so cannot collide with an existing
  filename. PyPI distributions cannot be replaced.

If publication partially succeeds, keep the existing tag and GitHub Release
while maintainers resolve the failed artifact. Never reuse a published version
for different package contents.

## One-time repository setup

Install a GitHub App on `Blaizzy/mlx-audio` with read/write access to Contents,
Issues, and Pull requests. Configure:

- Repository variable `RELEASE_BOT_CLIENT_ID`
- Repository secret `RELEASE_BOT_PRIVATE_KEY`

Create a GitHub environment named `pypi`, then register a PyPI Trusted Publisher
for repository `Blaizzy/mlx-audio`, workflow `release.yml`, and environment
`pypi`. Apply environment protection rules appropriate for a production package
publisher.

After the first successful automated release, revoke the old
`PYPI_API_TOKEN`.
