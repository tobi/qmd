import type { Database } from "./db.js";

export type Track = { name: string; phrase: string; collection: string };
type Match = { line: number; text: string };
export type MatchChange = { file: string; removed: Match[]; added: Match[] };

export function listTracks(db: Database): Track[] {
  return db.prepare("SELECT name, phrase, collection FROM tracked_searches ORDER BY name").all<Track>();
}

export function removeTrack(db: Database, name: string): void {
  if (!db.prepare("DELETE FROM tracked_searches WHERE name = ?").run(name).changes) {
    throw new Error(`Unknown tracked search: ${name}`);
  }
}

// Cancel equal occurrences in order. Inserting an unrelated line above a match
// changes its line number, but should not make that match look new.
function unmatched(lines: Match[], against: Match[]): Match[] {
  const counts = new Map<string, number>();
  for (const { text } of against) counts.set(text, (counts.get(text) ?? 0) + 1);
  return lines.filter(({ text }) => {
    const remaining = counts.get(text) ?? 0;
    if (!remaining) return true;
    counts.set(text, remaining - 1);
    return false;
  });
}

// The caller holds an immediate transaction so all paths see the same index
// version and two consumers cannot advance the same baseline concurrently.
function scan(db: Database, track: Track, baseline: boolean): MatchChange[] {
  const { name, phrase, collection } = track;
  const changes: MatchChange[] = [];
  const dirty = db.prepare(`
    SELECT d.path, d.hash, s.matches
    FROM documents d LEFT JOIN tracked_search_files s ON s.name = ? AND s.path = d.path
    WHERE d.active = 1 AND d.collection = ? AND (s.hash IS NULL OR s.hash != d.hash)
    ORDER BY d.path
  `).all<{ path: string; hash: string; matches: string | null }>(name, collection);
  const content = db.prepare("SELECT doc FROM content WHERE hash = ?");
  const save = db.prepare(`
    INSERT INTO tracked_search_files (name, path, hash, matches) VALUES (?, ?, ?, ?)
    ON CONFLICT(name, path) DO UPDATE SET hash = excluded.hash, matches = excluded.matches
  `);
  for (const file of dirty) {
    const body = content.get<{ doc: string }>(file.hash)!.doc;
    const current: Match[] = [];
    if (body.includes(phrase)) {
      for (const [i, text] of body.split(/\r\n|\n|\r/).entries()) {
        if (text.includes(phrase)) current.push({ line: i + 1, text });
      }
    }
    if (!baseline) {
      const previous: Match[] = file.matches ? JSON.parse(file.matches) : [];
      const removed = unmatched(previous, current);
      const added = unmatched(current, previous);
      if (added.length || removed.length) changes.push({ file: file.path, removed, added });
    }
    save.run(name, file.path, file.hash, JSON.stringify(current));
  }

  const gone = db.prepare(`
    SELECT path, matches FROM tracked_search_files s WHERE name = ? AND NOT EXISTS
      (SELECT 1 FROM documents d WHERE d.collection = ? AND d.path = s.path AND d.active = 1)
    ORDER BY path
  `).all<{ path: string; matches: string }>(name, collection);
  const drop = db.prepare("DELETE FROM tracked_search_files WHERE name = ? AND path = ?");
  for (const { path, matches } of gone) {
    const removed: Match[] = JSON.parse(matches);
    if (removed.length) changes.push({ file: path, removed, added: [] });
    drop.run(name, path);
  }
  return changes;
}

export function addTrack(db: Database, track: Track): void {
  if (!track.name.trim() || !track.phrase.trim() || /[\r\n]/.test(track.phrase)) {
    throw new Error("Supply a name and a nonempty, single-line phrase");
  }
  db.transaction(() => {
    const inserted = db.prepare(`
      INSERT INTO tracked_searches (name, phrase, collection) VALUES (?, ?, ?)
      ON CONFLICT(name) DO NOTHING
    `).run(track.name, track.phrase, track.collection);
    if (!inserted.changes) throw new Error(`Tracked search already exists: ${track.name}`);
    scan(db, track, true);
  }).immediate();
}

export function trackChanges(db: Database, name: string): { collection: string; changes: MatchChange[] } {
  return db.transaction(() => {
    const track = db.prepare("SELECT name, phrase, collection FROM tracked_searches WHERE name = ?").get<Track>(name);
    if (!track) throw new Error(`Unknown tracked search: ${name}`);
    return { collection: track.collection, changes: scan(db, track, false) };
  }).immediate();
}
