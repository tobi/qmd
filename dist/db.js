/**
 * db.ts - Cross-runtime SQLite compatibility layer
 *
 * Provides a unified Database export that works under both Bun (bun:sqlite)
 * and Node.js (better-sqlite3). The APIs are nearly identical — the main
 * difference is the import path.
 *
 * On macOS, Apple's system SQLite is compiled with SQLITE_OMIT_LOAD_EXTENSION,
 * which prevents loading native extensions like sqlite-vec. When running under
 * Bun we call Database.setCustomSQLite() to swap in Homebrew's full-featured
 * SQLite build before creating any database instances.
 */
export const isBun = typeof globalThis.Bun !== "undefined";
let _Database;
let _sqliteVecLoad;
if (isBun) {
    // Dynamic string prevents tsc from resolving bun:sqlite on Node.js builds
    const bunSqlite = "bun:" + "sqlite";
    const BunDatabase = (await import(/* @vite-ignore */ bunSqlite)).Database;
    // See: https://bun.com/docs/runtime/sqlite#setcustomsqlite
    if (process.platform === "darwin") {
        const homebrewPaths = [
            "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib", // Apple Silicon
            "/usr/local/opt/sqlite/lib/libsqlite3.dylib", // Intel
        ];
        for (const p of homebrewPaths) {
            try {
                BunDatabase.setCustomSQLite(p);
                break;
            }
            catch { }
        }
    }
    _Database = BunDatabase;
    // setCustomSQLite may have silently failed — test that extensions actually work.
    try {
        const { getLoadablePath } = await import("sqlite-vec");
        const vecPath = getLoadablePath();
        const testDb = new BunDatabase(":memory:");
        testDb.loadExtension(vecPath);
        testDb.close();
        _sqliteVecLoad = (db) => db.loadExtension(vecPath);
    }
    catch {
        // Vector search won't work, but BM25 and other operations are unaffected.
        _sqliteVecLoad = null;
    }
}
else {
    _Database = (await import("better-sqlite3")).default;
    const sqliteVec = await import("sqlite-vec");
    _sqliteVecLoad = (db) => sqliteVec.load(db);
}
/**
 * Open a SQLite database. Works with both bun:sqlite and better-sqlite3.
 */
export function openDatabase(path) {
    return new _Database(path);
}
/**
 * Load the sqlite-vec extension into a database.
 *
 * Throws with platform-specific fix instructions when the extension is
 * unavailable.
 */
export function loadSqliteVec(db) {
    if (!_sqliteVecLoad) {
        const hint = isBun && process.platform === "darwin"
            ? "On macOS with Bun, install Homebrew SQLite: brew install sqlite\n" +
                "Or install qmd with npm instead: npm install -g @tobilu/qmd"
            : "Ensure the sqlite-vec native module is installed correctly.";
        throw new Error(`sqlite-vec extension is unavailable. ${hint}`);
    }
    _sqliteVecLoad(db);
}
