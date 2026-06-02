"""UNC Chapel Hill (Economics, BS) ingestion into CAP-Bench v2.1.

Scoping (locked Phase INGEST-4): ECON courses with number < 500 (undergraduate;
graduate 500+ dropped) plus the MATH/STOR/COMP support courses referenced by the
BS programme in extracted_program.json.

Module: AI503. British English.
"""

from .real_ingest import InstConfig, ingest

CONFIG = InstConfig(
    university="unc",
    programme_id="UNC_BS_ECON_v2024",
    programme_name="Bachelor of Science in Economics",
    primary_department="ECON",
    home_subjects={"ECON"},
    major_subject="ECON",
    support_subjects={"MATH", "STOR", "COMP"},
    home_max_number=500,
    digits=3,
    native_total_credits=120,
    native_total_credits_note=(
        "Standard UNC bachelor's degree total (120 semester hours); the full "
        "degree includes the IDEAs-in-Action general-education curriculum and "
        "free electives outside the scoped Economics subset. Source: "
        "catalog.unc.edu BS Economics programme page."),
)


def main() -> None:
    ingest(CONFIG)


if __name__ == "__main__":
    main()
