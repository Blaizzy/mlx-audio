#!/usr/bin/env python3
import re
import sys

VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-rc(0|[1-9]\d*))?$")


def parse(value: str) -> tuple[tuple[int, int, int], int | None]:
    match = VERSION.fullmatch(value)
    if not match:
        raise ValueError(
            f"{value} is not a supported version; use MAJOR.MINOR.PATCH[-rcN]"
        )
    major, minor, patch, release_candidate = match.groups()
    return (int(major), int(minor), int(patch)), (
        int(release_candidate) if release_candidate is not None else None
    )


def validate(current_text: str, requested_text: str) -> None:
    current, current_rc = parse(current_text)
    requested, requested_rc = parse(requested_text)

    if requested < current:
        raise ValueError(f"{requested_text} is older than the current {current_text}")
    if requested == current:
        if current_rc is None:
            raise ValueError(
                f"{requested_text} is not newer than the current {current_text}"
            )
        if requested_rc is not None and requested_rc <= current_rc:
            raise ValueError(
                f"{requested_text} is not newer than the current {current_text}"
            )
        return

    major, minor, patch = current
    allowed = {
        (major, minor, patch + 1),
        (major, minor + 1, 0),
        (major + 1, 0, 0),
    }
    if requested not in allowed:
        choices = ", ".join(".".join(map(str, version)) for version in sorted(allowed))
        raise ValueError(
            f"{requested_text} skips a release line from {current_text}; "
            f"choose {choices}, optionally with -rcN"
        )


if __name__ == "__main__":
    try:
        validate(sys.argv[1], sys.argv[2])
    except (IndexError, ValueError) as error:
        raise SystemExit(f"release version rejected: {error}") from error
    print(f"release version accepted: {sys.argv[1]} -> {sys.argv[2]}")
