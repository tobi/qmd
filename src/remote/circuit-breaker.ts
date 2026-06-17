/**
 * circuit-breaker.ts — Fault-isolating circuit breaker for remote API calls.
 *
 * ## States
 *
 * - **closed**: Normal operation. Requests proceed and are tracked.
 * - **open**: Too many consecutive failures. Requests are rejected immediately
 *   with a descriptive error. Auto-recovers after cooldownMs.
 * - **half-open**: Cooldown has elapsed. One probe request is allowed through.
 *   If it succeeds, the breaker resets to closed. If it fails, back to open.
 *
 * ## Configuration
 *
 * - maxFailures: Consecutive failures before opening (default 3)
 * - cooldownMs: Time in open state before transitioning to half-open (default 10 min)
 *
 * ## Usage
 *
 * Each endpoint (embed, rerank, expand) gets its own breaker instance so
 * a failing embed endpoint doesn't block reranking and vice versa.
 *
 * @module remote/circuit-breaker
 */

export type CircuitState = 'closed' | 'open' | 'half-open';

export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures = 0;
  private lastFailureTime = 0;
  private halfOpenProbeInFlight = false;
  private readonly maxFailures: number;
  private readonly cooldownMs: number;

  /**
   * @param maxFailures - Consecutive failures before opening (default 3)
   * @param cooldownMs  - Cooldown duration in ms (default 10 minutes)
   */
  constructor(maxFailures = 3, cooldownMs = 10 * 60 * 1000) {
    this.maxFailures = maxFailures;
    this.cooldownMs = cooldownMs;
  }

  /**
   * Check if a request should be attempted.
   *
   * @returns true if the request can proceed, false if it should be rejected
   */
  canAttempt(): boolean {
    if (this.state === 'closed') return true;
    if (this.state === 'open') {
      if (Date.now() - this.lastFailureTime >= this.cooldownMs) {
        this.state = 'half-open';
        this.halfOpenProbeInFlight = true;
        return true;
      }
      return false;
    }
    // half-open: allow only one in-flight probe request.
    if (this.halfOpenProbeInFlight) return false;
    this.halfOpenProbeInFlight = true;
    return true;
  }

  /** Report a successful request. Resets the breaker to closed. */
  onSuccess(): void {
    this.state = 'closed';
    this.failures = 0;
    this.halfOpenProbeInFlight = false;
  }

  /** Report a failed request. May transition to open state. */
  onFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();
    this.halfOpenProbeInFlight = false;
    if (this.state === 'half-open' || this.failures >= this.maxFailures) {
      this.state = 'open';
    }
  }

  /** Get the current circuit state for diagnostics. */
  getState(): CircuitState {
    return this.state;
  }
}
