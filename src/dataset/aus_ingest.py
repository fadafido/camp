"""American University of Sharjah (Information Systems & Business Analytics)
ingestion into CAP-Bench v2.1.

Scoping (locked Phase INGEST-4): all undergraduate courses (number < 500) in the
business-core subjects ISA/ACC/FIN/MGT/MKT/ECO/STA/QBA/BLW, plus any course
referenced by the IS major in extracted_program.json (e.g. CMP 320, COE 420,
EGM 362, SCM 310/311, UPL 302). ISA is the major subject; the other business
subjects and referenced extras form the SUPPORT block.

Module: AI503. British English.
"""

from .real_ingest import InstConfig, ingest

CONFIG = InstConfig(
    university="aus",
    programme_id="AUS_BSBA_IS_v2024",
    programme_name=("Bachelor of Science in Business Administration, "
                    "Major in Information Systems and Business Analytics"),
    primary_department="ISA",
    home_subjects={"ISA", "ACC", "FIN", "MGT", "MKT", "ECO", "STA", "QBA", "BLW"},
    major_subject="ISA",
    support_subjects=set(),  # non-ISA scoping handled via home_subjects + referenced
    home_max_number=500,
    digits=3,
    keep_any_referenced=True,
    native_total_credits=120,
    native_total_credits_note=(
        "Standard AUS BSBA degree total (120 credits); the full degree includes "
        "the general-education programme and free electives outside the scoped "
        "Information-Systems subset. Source: AUS 2024-2025 undergraduate "
        "catalogue (PDF)."),
)


def main() -> None:
    ingest(CONFIG)


if __name__ == "__main__":
    main()
