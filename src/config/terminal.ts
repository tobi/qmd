/**
 * Terminal-specific configuration and utilities
 */

/**
 * Progress bar helper (Windows Terminal / VSCode-compatible)
 * Uses escape codes only when stderr is a TTY
 */
export const progress = {
  set(percent: number) {
    if (process.stderr.isTTY) {
      process.stderr.write(`\x1b]9;4;1;${Math.round(percent)}\x07`);
    }
  },
  clear() {
    if (process.stderr.isTTY) {
      process.stderr.write(`\x1b]9;4;0\x07`);
    }
  },
  indeterminate() {
    if (process.stderr.isTTY) {
      process.stderr.write(`\x1b]9;4;3\x07`);
    }
  },
  error() {
    if (process.stderr.isTTY) {
      process.stderr.write(`\x1b]9;4;2\x07`);
    }
  },
};
