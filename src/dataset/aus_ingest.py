"""American University of Sharjah (Information Systems & Business Analytics)
ingestion into CAP-Bench v2.1.

Scoping: the BSBA **Information Systems & Business Analytics MAJOR** — ISA is the
home (major) subject (its undergraduate courses, number < 500), plus ONLY the
specific business-core support courses the degree requires (ACC/FIN/MGT/MKT/ECO/
STA/QBA/BLW/SCM courses listed in the catalogue's BSBA business core, plus the
ISA prerequisite closure), as enumerated in extracted_program.json's
``referenced_course_codes``. This deliberately does NOT scope the whole business
school — only the IS major + its required core — so the scoped scale matches a
single ~120-credit major (consistent with Khalifa COSC and UNC ECON), rather than
all nine business subjects.

Module: AI503. British English.
"""

from .real_ingest import InstConfig, ingest

CONFIG = InstConfig(
    university="aus",
    programme_id="AUS_BSBA_IS_v2024",
    programme_name=("Bachelor of Science in Business Administration, "
                    "Major in Information Systems and Business Analytics"),
    primary_department="ISA",
    home_subjects={"ISA"},  # ISA is the major; all ISA undergrad courses are "home"
    major_subject="ISA",
    support_subjects=set(),  # non-ISA academic courses kept via referenced_course_codes
    home_max_number=500,
    digits=3,
    # referenced_course_codes is the REAL BSBA-IS academic scope (Business Core +
    # I&E + Major Requirements incl. "or" alternatives + Major Electives pool +
    # prerequisite closure); keep exactly those, regardless of subject.
    keep_any_referenced=True,
    native_total_credits=123,  # real BSBA degree total (catalogue-authoritative)
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
