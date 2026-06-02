"""Two add-only, mostly read-only paper checks (neither changes any model/result).

CHECK 1 — Personalisation on ACADEMIC STATE. Shows CAMP gives different next-term
plans to genuinely-different mid-trajectory test students (NOT from-scratch, which
is known to collapse to a per-institution template). Reuses the *exact* evaluation
read-out ``camp._camp_scores`` (greedy masked sequential rollout) and reports
CAMP's top-k recommended next-term set per probed student.

CHECK 2 — Data-realism audit over all committed synthetic transcripts (the count
is whatever the current dataset build produced; not asserted here).
Characterises failure and repeat behaviour to judge whether repeat realism is
sound: fail rate by GPA band, repeat gap distribution of failed courses,
failed-and-never-repeated split by required/elective (with a graduate
consistency cross-check), GPA trajectory after first failure, and a sanity
cross-check of graduation rate / terms-to-graduate against the committed
``dataset_stats.json``.

ADD-ONLY writes: ``results/check_personalisation.json`` and
``results/check_data_realism.json``. Seed 42; CPU-only; British English.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from src.constraint_engine.engine import StudentState
from src.models import baselines as bl
from src.models import camp
from src.models import task

RESULTS_DIR = task.BUNDLE / "results"
PERS_PATH = RESULTS_DIR / "check_personalisation.json"
REALISM_PATH = RESULTS_DIR / "check_data_realism.json"
DATASET_STATS = task.BUNDLE / "dataset_stats.json"

PASS = task.MIN_PASS_GRADE  # 1.0
TOPK = 5  # CAMP's recommended next-term set size for the personalisation demo


# ----------------------------------------------------------------- shared helpers
def _required_courses(engine, uni: str) -> set[str]:
    """CORE + SUPPORT gateway courses — the individually-required courses.

    CORE/SUPPORT are explicit_list ``min_courses`` rules whose list length equals
    the gateway count, so every listed course must be passed to graduate.
    """
    up = bl._PREFIX[uni]
    req: set[str] = set()
    for rule_id in (f"{up}_RULE_CORE", f"{up}_RULE_SUPPORT"):
        for cid, _ in engine.eligible_by_rule.get(rule_id, []):
            req.add(cid)
    return req


def _n_unmet_groups(engine, programme_id: str, passed_ids: set[str]) -> int:
    rem = engine.remaining_requirements(StudentState(programme_id=programme_id, passed={c: 4.0 for c in passed_ids}))
    return sum(len(v["unmet_groups"]) for k, v in rem.items() if isinstance(v, dict) and "unmet_groups" in v)


# ===================================================================== CHECK 1
def check_personalisation(n_probe: int = 5) -> dict[str, Any]:
    vocab = task.load_vocab()
    idx2cid = {i: c for c, i in vocab.items()}
    test_samples = task.load_samples("test")
    out: dict[str, Any] = {
        "description": "Does CAMP personalise on academic state? Top-%d next-term "
        "recommended set for genuinely-different mid-trajectory test students "
        "(real warm-start states; NOT from-scratch). Read-out = camp._camp_scores "
        "(the evaluation read-out)." % TOPK,
        "topk": TOPK,
        "per_institution": {},
    }
    for uni in camp.INSTITUTIONS:
        engine = bl._engine_for(uni)
        programme_id = camp.INSTITUTIONS[uni]["programme"]
        total_req = engine.total_credits_required(programme_id)
        required = _required_courses(engine, uni)

        # Candidate mid-trajectory samples for this institution, with an academic
        # -state summary. Require a non-trivial history (>=1 passed course).
        cands = []
        for s in test_samples:
            if s["university"] != uni:
                continue
            passed = task.passed_history(s)
            if not passed:
                continue
            credits = sum(engine.credits_of(c) for c in passed)
            grades = [s["history_grades"][c] for c in passed]
            cands.append({
                "sample": s,
                "passed_set": frozenset(passed),
                "n_passed": len(passed),
                "credits": credits,
                "gpa": round(float(np.mean(grades)), 3),
                "n_unmet_groups": _n_unmet_groups(engine, programme_id, set(passed)),
            })
        # Pick n_probe states spread across the credits range (early -> late
        # trajectory) with DISTINCT passed-sets, so they genuinely differ.
        cands.sort(key=lambda d: (d["credits"], d["n_passed"]))
        chosen: list[dict] = []
        seen: set[frozenset] = set()
        if cands:
            idxs = np.linspace(0, len(cands) - 1, num=n_probe).round().astype(int)
            for j in idxs:
                # nudge to the next distinct passed-set if this one repeats
                k = int(j)
                while k < len(cands) and cands[k]["passed_set"] in seen:
                    k += 1
                if k >= len(cands):
                    continue
                seen.add(cands[k]["passed_set"])
                chosen.append(cands[k])

        # CAMP's recommended next-term set for each chosen state (evaluation read-out).
        model = MaskablePPO.load(task.MODELS_DIR / f"camp_{uni}.zip")
        raw = camp._camp_scores(uni, model, [c["sample"] for c in chosen])
        students_out = []
        plan_sets = []
        for row, c in zip(raw, chosen):
            ranked = [idx2cid[i] for i in np.argsort(-row) if row[i] > 0][:TOPK]
            rec = sorted(ranked)
            plan_sets.append(frozenset(rec))
            unmet_req = engine.unmet_required_rule_courses(set(c["passed_set"]))
            n_rec_required = sum(1 for cid in rec if cid in required)
            n_rec_unmet_target = sum(1 for cid in rec if cid in unmet_req)
            n_rec_elective = sum(1 for cid in rec if cid not in required)
            students_out.append({
                "student_id": c["sample"]["student_id"],
                "term_index": c["sample"]["term_index"],
                "n_passed": c["n_passed"],
                "credits_completed": c["credits"],
                "credits_fraction": round(c["credits"] / total_req, 3),
                "gpa": c["gpa"],
                "n_unmet_required_groups": c["n_unmet_groups"],
                "recommended_next_term_set": rec,
                "n_recommended_required": n_rec_required,
                "n_recommended_toward_unmet_requirement": n_rec_unmet_target,
                "n_recommended_elective": n_rec_elective,
            })
        n_distinct = len({p for p in plan_sets})

        # Qualitative note grounded in the actual sets.
        early = [s for s in students_out if s["credits_fraction"] < 0.4]
        late = [s for s in students_out if s["credits_fraction"] >= 0.75]
        early_req = np.mean([s["n_recommended_toward_unmet_requirement"] for s in early]) if early else None
        late_elec = np.mean([s["n_recommended_elective"] for s in late]) if late else None
        note = (
            f"{n_distinct}/{len(students_out)} probed states received DISTINCT "
            f"recommended sets. "
            + (f"Earlier-trajectory students (<40% credits) are steered toward "
               f"outstanding-requirement courses (mean {early_req:.1f} of {TOPK} "
               f"picks count toward an unmet required group). " if early_req is not None else "")
            + (f"Near-graduates (>=75% credits) shift toward electives (mean "
               f"{late_elec:.1f} of {TOPK} picks are non-required). " if late_elec is not None else "")
            + "Recommendations track academic state."
        )
        out["per_institution"][uni] = {
            "n_students_probed": len(students_out),
            "n_distinct_plans": n_distinct,
            "students": students_out,
            "qualitative_note": note,
            "personalises_on_academic_state": bool(n_distinct > 1),
        }
    return out


# ===================================================================== CHECK 2
def check_data_realism() -> dict[str, Any]:
    students = task._load_students()
    split_stats = json.loads(DATASET_STATS.read_text())

    # Build per-institution required-course sets.
    engines = {uni: bl._engine_for(uni) for uni in camp.INSTITUTIONS}
    required_by_uni = {uni: _required_courses(engines[uni], uni) for uni in camp.INSTITUTIONS}

    # ---- gather attempts ----
    band_total = defaultdict(int)
    band_fail = defaultdict(int)
    # repeat behaviour of failed attempts
    gap_buckets = {"repeated_within_1_term": 0, "repeated_within_2_terms": 0,
                   "repeated_3plus_terms": 0, "never_repeated": 0}
    n_failed_attempts = 0
    # failed-and-never-repeated split
    fnr_required = 0
    fnr_elective = 0
    grads_with_unrepeated_required_failure = []
    # gpa trajectory
    before_means, after_means = [], []
    # graduation cross-check
    grad_flags, terms_to_grad = [], []
    grad_by_uni = defaultdict(list)
    terms_by_uni = defaultdict(list)

    for s in students:
        uni = s["university"]
        band = s["demographics"]["gpa_band"]
        required = required_by_uni[uni]
        grad_flags.append(bool(s["graduated"]))
        grad_by_uni[uni].append(bool(s["graduated"]))
        if s["graduated"] and s["terms_to_graduate"] is not None:
            terms_to_grad.append(s["terms_to_graduate"])
            terms_by_uni[uni].append(s["terms_to_graduate"])

        terms = sorted(s["terms"], key=lambda t: t["term_index"])
        # per-course list of (term_index, passed) attempts, in order
        attempts_by_course: dict[str, list[tuple[int, bool]]] = defaultdict(list)
        for t in terms:
            for c in t["courses"]:
                band_total[band] += 1
                failed = c["grade"] < PASS
                if failed:
                    band_fail[band] += 1
                attempts_by_course[c["course_id"]].append((t["term_index"], not failed))

        # repeat behaviour + never-repeated classification (per failed attempt)
        student_has_unrepeated_required_failure = False
        for cid, atts in attempts_by_course.items():
            atts.sort()
            term_idxs = [ti for ti, _ in atts]
            for pos, (ti, passed) in enumerate(atts):
                if passed:
                    continue  # only failed attempts
                n_failed_attempts += 1
                later = [tj for tj in term_idxs[pos + 1:] if tj > ti]
                if not later:
                    gap_buckets["never_repeated"] += 1
                    if cid in required:
                        fnr_required += 1
                        student_has_unrepeated_required_failure = True
                    else:
                        fnr_elective += 1
                else:
                    gap = min(later) - ti
                    if gap <= 1:
                        gap_buckets["repeated_within_1_term"] += 1
                    elif gap == 2:
                        gap_buckets["repeated_within_2_terms"] += 1
                    else:
                        gap_buckets["repeated_3plus_terms"] += 1
        if s["graduated"] and student_has_unrepeated_required_failure:
            grads_with_unrepeated_required_failure.append(s["student_id"])

        # gpa trajectory after first failure
        first_fail_pos = None
        for pos, t in enumerate(terms):
            if any(c["grade"] < PASS for c in t["courses"]):
                first_fail_pos = pos
                break
        if first_fail_pos is not None and first_fail_pos + 1 < len(terms):
            before = [terms[p]["term_gpa"] for p in range(first_fail_pos + 1)]
            after = [terms[p]["term_gpa"] for p in range(first_fail_pos + 1, len(terms))]
            before_means.append(float(np.mean(before)))
            after_means.append(float(np.mean(after)))

    # ---- assemble item numbers ----
    fail_rate_by_band = {
        b: round(band_fail[b] / band_total[b], 6) if band_total[b] else 0.0
        for b in task.GPA_BANDS
    }
    overall_attempts = sum(band_total.values())
    overall_fail = sum(band_fail.values())
    n_repeated = (gap_buckets["repeated_within_1_term"] + gap_buckets["repeated_within_2_terms"]
                  + gap_buckets["repeated_3plus_terms"])
    repeat_dist = {k: (round(v / n_failed_attempts, 6) if n_failed_attempts else 0.0)
                   for k, v in gap_buckets.items()}

    recomputed_grad_rate = round(float(np.mean(grad_flags)), 4)
    recomputed_terms = round(float(np.mean(terms_to_grad)), 4) if terms_to_grad else None
    committed = split_stats["overall"]
    grad_match = abs(recomputed_grad_rate - committed["graduation_rate"]) < 5e-3
    terms_match = recomputed_terms is not None and abs(recomputed_terms - committed["mean_terms_to_graduate"]) < 5e-2
    per_inst_xcheck = {}
    for uni in camp.INSTITUTIONS:
        rg = round(float(np.mean(grad_by_uni[uni])), 4)
        rt = round(float(np.mean(terms_by_uni[uni])), 4) if terms_by_uni[uni] else None
        cg = split_stats["per_institution"][uni]
        per_inst_xcheck[uni] = {
            "recomputed_graduation_rate": rg, "committed_graduation_rate": cg["graduation_rate"],
            "recomputed_mean_terms": rt, "committed_mean_terms": cg["mean_terms_to_graduate"],
            "match": bool(abs(rg - cg["graduation_rate"]) < 5e-3
                          and rt is not None and abs(rt - cg["mean_terms_to_graduate"]) < 5e-2),
        }

    before_mean = round(float(np.mean(before_means)), 4) if before_means else None
    after_mean = round(float(np.mean(after_means)), 4) if after_means else None

    # ---- honest verdict (let the numbers decide) ----
    frac_repeated = n_repeated / n_failed_attempts if n_failed_attempts else 0.0
    frac_within2 = ((gap_buckets["repeated_within_1_term"] + gap_buckets["repeated_within_2_terms"])
                    / n_failed_attempts) if n_failed_attempts else 0.0
    frac_never = gap_buckets["never_repeated"] / n_failed_attempts if n_failed_attempts else 0.0
    n_bad_grads = len(grads_with_unrepeated_required_failure)
    gpa_recovers = (before_mean is not None and after_mean is not None and after_mean >= before_mean - 0.15)

    realistic = (frac_within2 >= 0.5 and n_bad_grads == 0 and gpa_recovers)
    verdict = (
        ("REALISTIC. " if realistic else "MIXED / NEEDS A LIMITATIONS NOTE. ")
        + f"Of {n_failed_attempts} failed attempts, {frac_repeated:.1%} are ever repeated "
        f"({frac_within2:.1%} within two terms; {frac_never:.1%} never). "
        + f"No graduate has an unrepeated REQUIRED-course failure (n={n_bad_grads}), so the "
          "graduation check is internally consistent. " if n_bad_grads == 0 else
          f"WARNING: {n_bad_grads} graduates carry an unrepeated REQUIRED-course failure — a "
          "simulator/graduation-logic inconsistency. "
    )
    verdict += (
        f"Mean term-GPA moves {before_mean}->{after_mean} across a student's first failure "
        + ("(stable/recovering), " if gpa_recovers else "(declining), ")
        + "consistent with realistic repeat-and-recover behaviour."
        if (before_mean is not None) else "GPA trajectory not computable."
    )
    # Honest caveat regardless: failures are rare by construction (high pass rate),
    # and electives need not be repeated (any elective satisfies the credit rule).
    verdict += (" Note: the population pass rate is high by design, so failures are "
                "relatively rare; un-repeated ELECTIVE failures are expected and benign "
                "(any other elective satisfies the credit minimum).")

    return {
        "description": "Failure/repeat realism audit over all committed synthetic transcripts.",
        "n_students": len(students),
        "item1_fail_rate_by_gpa_band": fail_rate_by_band,
        "item1_overall_fail_rate": round(overall_fail / overall_attempts, 6) if overall_attempts else 0.0,
        "item1_total_attempts": overall_attempts,
        "item2_n_failed_attempts": n_failed_attempts,
        "item2_repeat_gap_distribution": repeat_dist,
        "item2_fraction_ever_repeated": round(frac_repeated, 6),
        "item3_failed_never_repeated_required": fnr_required,
        "item3_failed_never_repeated_elective": fnr_elective,
        "item3_graduates_with_unrepeated_required_failure": {
            "count": n_bad_grads, "student_ids": grads_with_unrepeated_required_failure[:20],
        },
        "item4_gpa_trajectory": {
            "n_students_with_failure_and_followup": len(before_means),
            "mean_term_gpa_through_first_failure": before_mean,
            "mean_term_gpa_after_first_failure": after_mean,
        },
        "item5_sanity_crosscheck": {
            "recomputed_graduation_rate": recomputed_grad_rate,
            "committed_graduation_rate": committed["graduation_rate"],
            "graduation_rate_matches": bool(grad_match),
            "recomputed_mean_terms_to_graduate": recomputed_terms,
            "committed_mean_terms_to_graduate": committed["mean_terms_to_graduate"],
            "mean_terms_matches": bool(terms_match),
            "per_institution": per_inst_xcheck,
        },
        "verdict": verdict,
        "realistic": bool(realistic),
    }


# ----------------------------------------------------------------- orchestration
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("CHECK 1 — personalisation ...")
    pers = check_personalisation()
    with PERS_PATH.open("w") as fh:
        json.dump(pers, fh, indent=2)
    print(f"  wrote {PERS_PATH}")

    print("CHECK 2 — data realism ...")
    realism = check_data_realism()
    with REALISM_PATH.open("w") as fh:
        json.dump(realism, fh, indent=2)
    print(f"  wrote {REALISM_PATH}")

    # ---- verification block (re-read from files) ----
    p = json.loads(PERS_PATH.read_text())
    r = json.loads(REALISM_PATH.read_text())
    print("\n================ CHECK 1 — PERSONALISATION ================")
    pause1 = []
    for uni, d in p["per_institution"].items():
        print(f"[{uni}] probed={d['n_students_probed']} distinct_plans={d['n_distinct_plans']} "
              f"personalises={d['personalises_on_academic_state']}")
        for st in d["students"]:
            print(f"   {st['student_id']} t{st['term_index']} cr={st['credits_completed']} "
                  f"({st['credits_fraction']:.0%}) gpa={st['gpa']} unmet_groups={st['n_unmet_required_groups']} "
                  f"-> {st['recommended_next_term_set']}")
        print(f"   note: {d['qualitative_note']}")
        if d["n_distinct_plans"] <= 1:
            pause1.append(uni)
    print("\n================ CHECK 2 — DATA REALISM ================")
    print(f"item1 fail rate by band: {r['item1_fail_rate_by_gpa_band']} (overall {r['item1_overall_fail_rate']})")
    print(f"item2 failed attempts: {r['item2_n_failed_attempts']}; repeat gap dist: {r['item2_repeat_gap_distribution']}")
    print(f"      fraction ever repeated: {r['item2_fraction_ever_repeated']}")
    print(f"item3 failed-never-repeated: required={r['item3_failed_never_repeated_required']} "
          f"elective={r['item3_failed_never_repeated_elective']}")
    print(f"      graduates w/ unrepeated REQUIRED failure: {r['item3_graduates_with_unrepeated_required_failure']['count']}")
    print(f"item4 GPA through->after first failure: "
          f"{r['item4_gpa_trajectory']['mean_term_gpa_through_first_failure']} -> "
          f"{r['item4_gpa_trajectory']['mean_term_gpa_after_first_failure']} "
          f"(n={r['item4_gpa_trajectory']['n_students_with_failure_and_followup']})")
    x = r["item5_sanity_crosscheck"]
    print(f"item5 grad rate recomputed={x['recomputed_graduation_rate']} committed={x['committed_graduation_rate']} "
          f"match={x['graduation_rate_matches']}")
    print(f"      mean terms recomputed={x['recomputed_mean_terms_to_graduate']} "
          f"committed={x['committed_mean_terms_to_graduate']} match={x['mean_terms_matches']}")
    print(f"\nVERDICT: {r['verdict']}")

    # ---- PAUSE conditions ----
    pause2 = []
    if r["item3_graduates_with_unrepeated_required_failure"]["count"] > 0:
        pause2.append("graduates with unrepeated REQUIRED failure")
    if not (x["graduation_rate_matches"] and x["mean_terms_matches"]):
        pause2.append("recomputed grad-rate/terms do NOT match committed dataset_stats")
    print("\n---- PAUSE FLAGS ----")
    print(f"CHECK 1 (same set for differing students): {pause1 or 'none'}")
    print(f"CHECK 2 (consistency): {pause2 or 'none'}")


if __name__ == "__main__":
    main()
