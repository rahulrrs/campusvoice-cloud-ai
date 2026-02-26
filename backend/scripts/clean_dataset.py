import os
import json
import re
import pandas as pd

# ================== CONFIG ==================
IN_PATH = r"data\dataset.csv"
OUT_PATH = r"data\dataset_clean.csv"
OUT_DIR = r"outputs\edu_classifier_multitask"
MIN_WORDS = 5

DEFAULT_PRIORITY_ID = 1  # 0=Low, 1=Medium, 2=High

# ── Label-noise fix: reclassify course-review texts under "Examination" → "Academic"
FIX_EXAM_LABEL_NOISE = True

# ── Synthetic augmentation for underrepresented High-priority exam complaints
AUGMENT_EXAM_HIGH = True
# ===========================================

# ─── Regex library ───────────────────────────────────────────────────────────
_URGENCY_RE = re.compile(
    r"\b(today|tomorrow|tonight|in\s*\d+\s*(hours|hrs)|within\s*\d+\s*(hours|hrs)|next\s*day)\b",
    re.I,
)
_EXAM_COMPLAINT_RE = re.compile(
    r"\b(hall\s*ticket|admit\s*card|timetable|time\s*table|seating|venue|result|revaluation|"
    r"registration|deadline|last\s*date|not\s*released|schedule|roll\s*no|seat\s*number|"
    r"exam\s*centre|exam\s*date)\b",
    re.I,
)
_COURSE_REVIEW_RE = re.compile(
    r"\b(prof|professor|lecture|course|assignment|midterm|mark|grading|\bTA\b|quiz|"
    r"textbook|readings?|semester|instructor|courseload|syllabus|coursework|"
    r"professor's?|lectures?|assignments?)\b",
    re.I,
)
_EXAM_BLOCKER_RE = re.compile(
    r"\b(exam|examination|timetable|time\s*table|schedule|hall\s*ticket|admit\s*card|result|"
    r"revaluation|registration|enroll|deadline|starts?\s*(tomorrow|today))\b",
    re.I,
)
_WATER_OUTAGE_RE = re.compile(
    r"\b(no\s*water|water\s*(is\s*)?not\s*available|water\s*supply\s*.{0,20}\bdown\b|"
    r"water\s*problem|water\s*outage|water\s*cut)\b",
    re.I,
)
_DURATION_RE = re.compile(r"\b(\d+)\s*(day|days|d)\b", re.I)


def _is_exam_label_noise(text: str) -> bool:
    """Return True if an 'Examination' row is actually a course review (not a real complaint)."""
    if _EXAM_COMPLAINT_RE.search(text):
        return False  # has real exam complaint keywords → keep as Examination
    return bool(_COURSE_REVIEW_RE.search(text))


def _is_exam_urgent(text: str) -> bool:
    t = (text or "").lower()
    if any(k in t for k in ["scholarship", "discount", "fee waiver", "fees", "financial aid"]):
        return False
    return bool(_EXAM_BLOCKER_RE.search(t) and _URGENCY_RE.search(t))


def _is_hostel_water_outage(text: str) -> bool:
    t = (text or "").lower()
    if not _WATER_OUTAGE_RE.search(t):
        return False
    m = _DURATION_RE.search(t)
    if not m:
        return True
    try:
        days = int(m.group(1))
    except ValueError:
        return True
    return days >= 1


def safe_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


# ─── Synthetic exam-complaint rows ───────────────────────────────────────────
_EXAM_SYNTHETIC = [
    "Hall ticket is not available on the portal and exam is tomorrow. Please resolve urgently.",
    "My admit card has not been released and the exam starts tomorrow. Need immediate help.",
    "Exam timetable has not been published and the exam begins tomorrow. Very stressful.",
    "Exam centre and seating details are not released. Exam is next day. Please update.",
    "Hall ticket not yet issued. Exam is today and I cannot appear without it.",
    "Result not published yet; revaluation deadline is tomorrow. Please act fast.",
    "Registration portal is down and the exam enrollment deadline is today.",
    "Timetable not released and exam is starting today. Please share the schedule.",
    "I have not received my hall ticket and the examination is tomorrow morning.",
    "Admit card is missing from the portal and I have an exam in a few hours.",
    "Seating arrangement and venue details for tomorrow's exam still not uploaded.",
    "Exam date is tomorrow and the admit card link on the portal is broken.",
    "Hall ticket download failing; exam hall is tomorrow at 9 AM. Urgent fix needed.",
    "The exam timetable was changed without notice and my hall ticket shows wrong date.",
    "Revaluation form not working and last date is today. Please fix immediately.",
]


def main():
    print(f"📥 Loading: {IN_PATH}")
    df = pd.read_csv(IN_PATH, low_memory=False)

    # --- required columns ---
    need = {"text", "label", "label_id", "priority_id_fixed"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    # --- clean text/labels ---
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["label"] = safe_strip(df["label"])

    if "priority_fixed" in df.columns:
        df["priority_fixed"] = safe_strip(df["priority_fixed"])
    if "priority" in df.columns:
        df["priority"] = safe_strip(df["priority"])

    # --- remove empty text ---
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    print(f"🧹 Removed empty: {before - len(df)}")

    # --- remove short text (< MIN_WORDS) ---
    before = len(df)
    df["word_count"] = df["text"].str.split().str.len().astype(int)
    df = df[df["word_count"] >= MIN_WORDS].copy()
    print(f"🧹 Removed short (<{MIN_WORDS} words): {before - len(df)}")

    # --- fix/cast label_id ---
    before = len(df)
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    print(f"🧹 Removed invalid label_id: {before - len(df)}")

    # ========================================================
    # ✅ FIX 1: EXAMINATION LABEL NOISE
    #    Course-review texts mislabeled as "Examination" are
    #    reclassified to "Academic" (label_id=0) so the model
    #    learns a clean boundary for real exam complaints.
    # ========================================================
    if FIX_EXAM_LABEL_NOISE:
        academic_label_id = int(df[df["label"] == "Academic"]["label_id"].iloc[0])
        exam_mask = df["label"] == "Examination"
        noise_mask = exam_mask & df["text"].apply(_is_exam_label_noise)
        reclassified = int(noise_mask.sum())
        df.loc[noise_mask, "label"] = "Academic"
        df.loc[noise_mask, "label_id"] = academic_label_id
        # Course reviews are typically Low priority – keep existing priority
        print(f"🔁 Reclassified Examination→Academic (label noise): {reclassified} rows")

    # ========================================================
    # ✅ FIX 2: PRIORITY HANDLING
    # ========================================================
    df["priority_id_fixed"] = pd.to_numeric(df["priority_id_fixed"], errors="coerce")

    pmap = {
        "low": 0, "l": 0, "0": 0,
        "medium": 1, "med": 1, "m": 1, "1": 1,
        "high": 2, "h": 2, "2": 2,
    }

    invalid_mask = df["priority_id_fixed"].isna() | (~df["priority_id_fixed"].isin([0, 1, 2]))

    recovered = 0
    if invalid_mask.any():
        src_col = (
            "priority_fixed" if "priority_fixed" in df.columns
            else ("priority" if "priority" in df.columns else None)
        )
        if src_col:
            tmp = df.loc[invalid_mask, src_col].astype(str).str.strip().str.lower()
            rec = tmp.map(pmap)
            rec_mask = rec.notna()
            df.loc[invalid_mask & rec_mask, "priority_id_fixed"] = rec[rec_mask].astype(int)
            recovered = int(rec_mask.sum())

    invalid_mask2 = df["priority_id_fixed"].isna() | (~df["priority_id_fixed"].isin([0, 1, 2]))
    imputed = int(invalid_mask2.sum())
    if imputed > 0:
        df.loc[invalid_mask2, "priority_id_fixed"] = DEFAULT_PRIORITY_ID

    df["priority_imputed"] = 0
    if imputed > 0:
        df.loc[invalid_mask2, "priority_imputed"] = 1

    df["priority_id_fixed"] = df["priority_id_fixed"].astype(int)

    print(f"🛠️  Recovered priority from text: {recovered}")
    print(f"🛠️  Imputed priority_id_fixed to {DEFAULT_PRIORITY_ID} (Medium) for rows: {imputed}")

    # ========================================================
    # ✅ FIX 3: HEURISTIC PRIORITY OVERRIDES
    #    Applied BEFORE augmentation so synthetic rows inherit
    #    correct ground-truth priorities.
    # ========================================================
    before_override = df["priority_id_fixed"].copy()

    exam_mask = df["text"].astype(str).map(_is_exam_urgent)
    water_mask = df["text"].astype(str).map(_is_hostel_water_outage)

    df.loc[exam_mask, "priority_id_fixed"] = 2
    df.loc[water_mask, "priority_id_fixed"] = 2

    overridden = int((before_override != df["priority_id_fixed"]).sum())
    print(f"⬆️  Priority overrides → High (exam/water urgency): {overridden}")

    # ========================================================
    # ✅ FIX 4: SYNTHETIC AUGMENTATION FOR HIGH-PRIORITY EXAM
    #    The 'Examination' + High-priority combination is very
    #    rare (<100 rows). Add targeted synthetic rows so the
    #    model can learn this important pattern.
    # ========================================================
    if AUGMENT_EXAM_HIGH:
        exam_label_id = int(df[df["label"] == "Examination"]["label_id"].iloc[0])
        synth_rows = []
        for t in _EXAM_SYNTHETIC:
            synth_rows.append({
                "text": t,
                "label": "Examination",
                "label_id": exam_label_id,
                "priority_id_fixed": 2,           # High
                "priority_imputed": 0,
                "word_count": len(t.split()),
                "text_len": len(t),
            })
        synth_df = pd.DataFrame(synth_rows)
        # Fill any extra cols with NaN so concat works cleanly
        for col in df.columns:
            if col not in synth_df.columns:
                synth_df[col] = None
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f"➕  Synthetic exam-High rows added: {len(synth_rows)}")

    # --- derived cols ---
    df["text_len"] = df["text"].str.len().astype(int)
    df["word_count"] = df["text"].str.split().str.len().astype(int)

    # --- duplicates + conflicts ---
    conflict = df.groupby("text")[["label_id", "priority_id_fixed"]].nunique().reset_index()
    conflicting = conflict[(conflict["label_id"] > 1) | (conflict["priority_id_fixed"] > 1)]
    print(f"⚠️  Conflicting texts (label or priority): {len(conflicting)}")

    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    print(f"🧹 Dropped duplicates: {before - len(df)}")

    # ── Summary ──────────────────────────────────────────────
    print("\n📊 Final label distribution:")
    print(df["label"].value_counts().to_string())
    print("\n📊 Final priority distribution:")
    print(df["priority_id_fixed"].value_counts().sort_index().to_string())
    print(f"\n📊 Examination + High rows: {len(df[(df['label']=='Examination') & (df['priority_id_fixed']==2)])}")

    # --- save cleaned dataset ---
    data_dir = os.path.dirname(OUT_PATH) or "."
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n💾 Saved cleaned dataset: {OUT_PATH}  rows: {len(df)}")

    # --- save mappings ---
    os.makedirs(OUT_DIR, exist_ok=True)

    id_to_label = (
        df[["label_id", "label"]]
        .drop_duplicates()
        .sort_values("label_id")
        .set_index("label_id")["label"]
        .to_dict()
    )

    DEFAULT_PRIO_MAP = {0: "Low", 1: "Medium", 2: "High"}

    if "priority_fixed" in df.columns:
        tmp = df[["priority_id_fixed", "priority_fixed"]].drop_duplicates().copy()
        tmp["priority_fixed"] = tmp["priority_fixed"].astype(str).str.strip()

        def _norm_prio(v):
            if v is None:
                return None
            s = str(v).strip().lower()
            if s in {"", "nan"}:
                return None
            if s in {"low", "l", "0"}:
                return "Low"
            if s in {"medium", "med", "m", "1"}:
                return "Medium"
            if s in {"high", "h", "2"}:
                return "High"
            return None

        tmp["priority_fixed"] = tmp["priority_fixed"].map(_norm_prio)
        id_to_priority = {
            int(k): v
            for k, v in (
                tmp.dropna(subset=["priority_fixed"])
                .sort_values("priority_id_fixed")
                .groupby("priority_id_fixed")["priority_fixed"]
                .first()
                .items()
            )
        }
    else:
        id_to_priority = {}

    for k, v in DEFAULT_PRIO_MAP.items():
        id_to_priority.setdefault(k, v)

    with open(os.path.join(OUT_DIR, "id_to_label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "id_to_priority.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_priority.items()}, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved mappings → {OUT_DIR}")


if __name__ == "__main__":
    main()