# Dataset Policy

YouthNaturalLoRA uses explicit rights lanes and never mixes lanes into an
ambiguous adapter.

## Release Lane

The releasable lane is limited to:

- official Mozilla Common Voice Scripted Speech English releases verified on
  Mozilla Data Collective, filtered locally to validated clips and provided
  `teens` metadata only;
- official Mozilla Common Voice Spontaneous Speech English releases verified on
  Mozilla Data Collective, used as conversational-style data only unless
  official metadata provides an age band;
- explicitly consented private recordings with documented speaker and guardian
  consent when minors are involved.

Common Voice data remains subject to no-reidentification and no-rehosting
constraints recorded in `artifacts/youth_natural/dataset_rights_report.json`.

## Restricted Lanes

MyST, CSLU Kids, CMU Kids, and Expresso are not release-lane data in this branch.
They are blocked unless an executed license permits the exact model-training and
adapter-release use. Noncommercial data must not contaminate a releasable adapter.

## Privacy Rules

The pipeline stores hashes, pseudonymous IDs, and aggregate metrics in Git. It
does not store raw minor audio, raw speaker embeddings, access tokens, identity
metadata, or private paths in tracked artifacts.
