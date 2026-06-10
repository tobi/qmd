/**
 * log.ts — Pluggable logger for remote modules.
 *
 * All remote functions accept an optional Logger so callers can silence,
 * redirect, or spy on log output. Defaults to console methods.
 *
 * @module remote/log
 */

/**
 * Logger interface used by all remote modules.
 *
 * Implementations: ConsoleLogger (default), NoopLogger (silent),
 * or custom (for test spies / structured logging).
 */
export interface Logger {
  debug(msg: string, ...args: unknown[]): void;
  info(msg: string, ...args: unknown[]): void;
  warn(msg: string, ...args: unknown[]): void;
  error(msg: string, ...args: unknown[]): void;
}

/** Default logger using console methods. */
export const consoleLogger: Logger = {
  debug: (...args) => console.debug(...args),
  info: (...args) => console.info(...args),
  warn: (...args) => console.warn(...args),
  error: (...args) => console.error(...args),
};

/** Silent logger that discards all messages. Useful for tests. */
export const silentLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};
