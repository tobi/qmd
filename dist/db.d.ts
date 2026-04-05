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
export declare const isBun: boolean;
/**
 * Open a SQLite database. Works with both bun:sqlite and better-sqlite3.
 */
export declare function openDatabase(path: string): Database;
/**
 * Common subset of the Database interface used throughout QMD.
 */
export interface Database {
    exec(sql: string): void;
    prepare(sql: string): Statement;
    loadExtension(path: string): void;
    close(): void;
}
export interface Statement {
    run(...params: any[]): {
        changes: number;
        lastInsertRowid: number | bigint;
    };
    get(...params: any[]): any;
    all(...params: any[]): any[];
}
/**
 * Load the sqlite-vec extension into a database.
 *
 * Throws with platform-specific fix instructions when the extension is
 * unavailable.
 */
export declare function loadSqliteVec(db: Database): void;
