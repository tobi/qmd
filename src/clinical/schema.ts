/**
 * Clinical front-matter schema.
 *
 * Defines the YAML fields parsed out of clinical note front-matter and stored
 * in the `document_metadata` sidecar table. See clinical/schema.md for the
 * full spec, sample, and SQL DDL.
 *
 * Implementation of the parser and the indexer wiring lands in M2.
 */

export type NoteType =
  | "SOAP"
  | "progress"
  | "H&P"
  | "consult"
  | "op-note"
  | "discharge"
  | "telehealth"
  | "followup"
  | "referral"
  | "imaging"
  | "path"
  | "other";

export type EncounterSetting =
  | "outpatient"
  | "inpatient"
  | "ed"
  | "telehealth"
  | "home"
  | "other";

export interface ClinicalFrontmatter {
  note_type?: NoteType;
  encounter_date?: string;
  patient_id?: string;
  provider?: string;
  specialty?: string;
  setting?: EncounterSetting;
  problem_list?: string[];
  icd_codes?: string[];
  snomed_codes?: string[];
  medications?: string[];
  labs_mentioned?: string[];
  procedures?: string[];
  allergies?: string[];
  tags?: string[];
}

export const CLINICAL_FRONTMATTER_FIELDS = [
  "note_type",
  "encounter_date",
  "patient_id",
  "provider",
  "specialty",
  "setting",
  "problem_list",
  "icd_codes",
  "snomed_codes",
  "medications",
  "labs_mentioned",
  "procedures",
  "allergies",
  "tags",
] as const;

export const DOCUMENT_METADATA_DDL = `
CREATE TABLE IF NOT EXISTS document_metadata (
  hash             TEXT PRIMARY KEY,
  note_type        TEXT,
  encounter_date   TEXT,
  patient_id       TEXT,
  provider         TEXT,
  specialty        TEXT,
  setting          TEXT,
  problem_list     TEXT,
  icd_codes        TEXT,
  snomed_codes     TEXT,
  medications      TEXT,
  labs_mentioned   TEXT,
  procedures       TEXT,
  allergies        TEXT,
  tags             TEXT,
  raw_frontmatter  TEXT,
  parsed_at        TEXT NOT NULL,
  FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_docmeta_note_type      ON document_metadata(note_type);
CREATE INDEX IF NOT EXISTS idx_docmeta_encounter_date ON document_metadata(encounter_date);
CREATE INDEX IF NOT EXISTS idx_docmeta_patient_id     ON document_metadata(patient_id);
CREATE INDEX IF NOT EXISTS idx_docmeta_specialty      ON document_metadata(specialty);
`;
