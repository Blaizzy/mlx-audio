from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import RIGHTS_LANES, ValidationError


@dataclass(frozen=True)
class RightsRecord:
    source: str
    exact_release: str
    acquisition_date: str
    license: str
    dataset_terms_url: str
    forbidden_uses: list[str]
    redistribution_rule: str
    commercial_use_status: str
    model_release_status: str
    consent_notes: str
    terms_hash: str
    rights_lane: str
    checked_date: str
    decision: str
    evidence_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_terms_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_rights_records(checked_date: str = "2026-06-19") -> list[RightsRecord]:
    return [
        RightsRecord(
            source="Mozilla Common Voice Scripted Speech English",
            exact_release="v26.0 / cv-corpus-26.0-2026-06-12",
            acquisition_date="not_acquired",
            license="CC0-1.0",
            dataset_terms_url="https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu",
            forbidden_uses=[
                "speaker re-identification",
                "dataset re-hosting or re-sharing outside allowed platform terms",
            ],
            redistribution_rule="Do not re-host or re-share the dataset in this project.",
            commercial_use_status="dataset license is permissive; platform/privacy terms still apply",
            model_release_status="allowed only for release-lane snapshots that exclude restricted data",
            consent_notes="Use provided self-declared age metadata only; do not infer age.",
            terms_hash=hash_terms_text(
                "Common Voice Scripted Speech 26.0 English CC0-1.0; no re-identification; no re-host/re-share."
            ),
            rights_lane="permissive_release",
            checked_date=checked_date,
            decision="allowed_with_constraints",
            evidence_urls=[
                "https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu",
                "https://github.com/common-voice/cv-dataset/",
            ],
        ),
        RightsRecord(
            source="Mozilla Common Voice Spontaneous Speech English",
            exact_release="v4.0 / sps-corpus-4.0-2026-06-12",
            acquisition_date="not_acquired",
            license="CC0-1.0",
            dataset_terms_url="https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk",
            forbidden_uses=[
                "speaker re-identification",
                "dataset re-hosting or re-sharing outside allowed platform terms",
            ],
            redistribution_rule="Do not re-host or re-share the dataset in this project.",
            commercial_use_status="dataset license is permissive; platform/privacy terms still apply",
            model_release_status="allowed only as conversational-style source, not youth-labeled unless metadata says so",
            consent_notes="Do not assume speakers are young without official metadata.",
            terms_hash=hash_terms_text(
                "Common Voice Spontaneous Speech 4.0 English CC0-1.0; no re-identification; no re-host/re-share."
            ),
            rights_lane="permissive_release",
            checked_date=checked_date,
            decision="allowed_with_constraints",
            evidence_urls=[
                "https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk",
                "https://github.com/common-voice/cv-dataset/",
            ],
        ),
        RightsRecord(
            source="MyST Children's Conversational Speech",
            exact_release="LDC2021S05",
            acquisition_date="not_acquired",
            license="LDC agreement, non-commercial/research by default",
            dataset_terms_url="https://catalog.ldc.upenn.edu/LDC2021S05",
            forbidden_uses=["commercial use without separate license", "unlicensed redistribution"],
            redistribution_rule="No redistribution from this project.",
            commercial_use_status="blocked unless separate Boulder Learning commercial license exists",
            model_release_status="research-only until counsel/user supplies executed license",
            consent_notes="Children's speech; keep local and private.",
            terms_hash=hash_terms_text("MyST LDC2021S05 noncommercial/research default"),
            rights_lane="research_noncommercial",
            checked_date=checked_date,
            decision="blocked_for_release_allowed_only_after_license_acceptance",
            evidence_urls=["https://catalog.ldc.upenn.edu/LDC2021S05"],
        ),
        RightsRecord(
            source="Expresso conversational speech",
            exact_release="project page current on 2026-06-19",
            acquisition_date="not_acquired",
            license="CC BY-NC 4.0 per rights audit",
            dataset_terms_url="https://speechbot.github.io/expresso/",
            forbidden_uses=["commercial use", "release-lane contamination"],
            redistribution_rule="Follow project license; do not include raw audio here.",
            commercial_use_status="noncommercial only",
            model_release_status="research-only, not release-lane",
            consent_notes="Small speaker count; cap sampling if used in private research.",
            terms_hash=hash_terms_text("Expresso CC BY-NC 4.0"),
            rights_lane="research_noncommercial",
            checked_date=checked_date,
            decision="blocked_for_release_research_only",
            evidence_urls=["https://speechbot.github.io/expresso/"],
        ),
    ]


def write_rights_report(path: str | Path, records: list[RightsRecord]) -> None:
    for record in records:
        if record.rights_lane not in RIGHTS_LANES:
            raise ValidationError(f"invalid rights lane: {record.rights_lane}")
    out = {
        "format_version": 1,
        "generated_at": "2026-06-19",
        "final_legal_decision": "not_provided",
        "records": [record.to_dict() for record in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

