# Clinical Front-Matter Schema

clinical-qmd parses clinical metadata from the YAML front-matter of each note file and stores it in a sidecar SQLite table (`document_metadata`). The base `documents` and `content` tables from upstream are not modified — clinical data is purely additive, which keeps upstream merges clean.

## Recognized YAML front-matter fields

Every field is optional. Missing fields are simply not indexed; they do not cause errors.

```yaml
---
note_type: progress          # SOAP | progress | H&P | consult | op-note | discharge | telehealth | followup | referral | imaging | path | other
encounter_date: 2026-05-22   # ISO 8601 date (date or datetime)
patient_id: PT-00482          # pseudo-ID. Never a real MRN. Never a name.
provider: Dr. O. Said         # free text
specialty: cardiology         # free text or SNOMED specialty code
setting: outpatient           # outpatient | inpatient | ed | telehealth | home | other

problem_list:                 # active problems addressed in this note
  - HTN
  - CKD stage 3
  - "Type 2 diabetes mellitus"

icd_codes:                    # ICD-10-CM
  - I10
  - N18.30
  - E11.9

snomed_codes:                 # SNOMED CT concept IDs
  - 38341003                  # HTN
  - 433144002                 # CKD stage 3

medications:                  # active meds at time of note
  - lisinopril 20 mg daily
  - metformin 1000 mg BID

labs_mentioned:               # labs referenced or ordered in this note
  - A1c
  - eGFR
  - BMP

procedures:                   # procedures performed in this note
  - 99214

allergies:                    # NKDA permitted
  - penicillin (rash)

tags:                         # free-form labels
  - chronic-care
  - case-of-the-week
---

# Note body in markdown follows...
```

## Sidecar SQL schema

```sql
CREATE TABLE IF NOT EXISTS document_metadata (
  hash             TEXT PRIMARY KEY,
  note_type        TEXT,
  encounter_date   TEXT,             -- ISO 8601
  patient_id       TEXT,             -- pseudo-ID only
  provider         TEXT,
  specialty        TEXT,
  setting          TEXT,
  problem_list     TEXT,             -- JSON array
  icd_codes        TEXT,             -- JSON array
  snomed_codes     TEXT,             -- JSON array
  medications      TEXT,             -- JSON array
  labs_mentioned   TEXT,             -- JSON array
  procedures       TEXT,             -- JSON array
  allergies        TEXT,             -- JSON array
  tags             TEXT,             -- JSON array
  raw_frontmatter  TEXT,             -- full original YAML, forward-compat
  parsed_at        TEXT NOT NULL,
  FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_docmeta_note_type      ON document_metadata(note_type);
CREATE INDEX IF NOT EXISTS idx_docmeta_encounter_date ON document_metadata(encounter_date);
CREATE INDEX IF NOT EXISTS idx_docmeta_patient_id     ON document_metadata(patient_id);
CREATE INDEX IF NOT EXISTS idx_docmeta_specialty      ON document_metadata(specialty);
```

JSON arrays (problem list, codes, meds, labs) are stored as JSON text and queried with SQLite's `json_each()` table-valued function when filtering by membership. This keeps the schema simple while letting us do queries like:

```sql
SELECT d.path
FROM documents d
JOIN document_metadata m USING (hash)
JOIN json_each(m.icd_codes) c
WHERE c.value = 'E11.9'
  AND m.encounter_date >= '2026-01-01'
ORDER BY m.encounter_date DESC;
```

## PHI posture for the schema

- `patient_id` is a **pseudo-ID** chosen by the physician (e.g. `PT-00482`). Real MRNs and names should not be written into front-matter — they belong in the note body where the physician already knows they are present.
- `provider` is intended to be the physician using the system, not the patient.
- Indexes are local-only. The SQLite file at `~/.cache/qmd/index.sqlite` should sit on an encrypted volume.

## Validation

A front-matter parsing failure should never block indexing of the note body. If front-matter is malformed:

1. Log a warning with the file path and the parse error.
2. Skip the metadata insert for that note.
3. Index the body normally so the note remains searchable by text.

This is deliberate: a physician's notes are not test fixtures and clinical-qmd should never refuse to index a real note because its YAML is slightly off.

## Open questions for M2

- Should `patient_id` be hashed at index time to make the sidecar table itself non-identifying? (Likely yes — easy to add and a meaningful defense-in-depth.)
- Should we support both an explicit `icd_codes:` list and inline `[I10]`-style annotations in the body? (Probably yes, with body-extraction as M3.)
- Should `problem_list` be deduplicated against ICD codes automatically? (No — let the physician keep both, they encode different intent.)
