"""Khalifa University (Computer Science) ingestion into CAP-Bench v2.1.

Scoping (locked Phase INGEST-4): all undergraduate COSC courses (number < 500)
plus support courses referenced by the BSc CS requirements page from
MATH/CHEM/PHYS/CCEN/ECCE/ENGR/GENS. See :mod:`src.dataset.real_ingest`.

Module: AI503. British English.
"""

from .real_ingest import InstConfig, ingest

CONFIG = InstConfig(
    university="khalifa",
    programme_id="KHAL_BSC_CS_v2024",
    programme_name="Bachelor of Science in Computer Science",
    primary_department="COSC",
    home_subjects={"COSC"},
    major_subject="COSC",
    support_subjects={"MATH", "CHEM", "PHYS", "CCEN", "ECCE", "ENGR", "GENS"},
    home_max_number=500,
    digits=3,
    native_total_credits=130,
    native_total_credits_note=(
        "BSc Computer Science total credit hours from the Khalifa University "
        "2024-2025 undergraduate catalogue requirements page; the full degree "
        "includes general-education/university requirements outside the scoped "
        "Computer-Science subset."),
)


def main() -> None:
    ingest(CONFIG)


if __name__ == "__main__":
    main()
