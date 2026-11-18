/**
 * legacy-adoption-stub-llm - the model stand-in for the legacy fingerprint
 * adoption memory workers.
 *
 * Adoption only tokenizes, detokenizes and embeds the sampled body, so those
 * workers never load a model. `Store.llm` holds a real LlamaCpp, so the stub is
 * a subclass rather than a look-alike object: it satisfies that type exactly
 * and still answers the three calls the adoption path makes, deterministically
 * - 16 characters per token, one `x` per character on the way back, and one
 * fixed vector for every chunk.
 */
import type { Token } from "node-llama-cpp";
import { LlamaCpp, type EmbeddingResult } from "../../src/llm.ts";

/** The model the stub reports, and the one the workers store vectors under. */
export const LEGACY_ADOPTION_STUB_MODEL = "model";

// A token id is node-llama-cpp's branded number: 1 is a valid id for this
// stub's tokenizer.
const stubToken = 1 as Token;

export class LegacyAdoptionStubLlm extends LlamaCpp {
  // Chunking takes its character-based branch only for remote models, so a
  // fixed local name keeps that decision - and the fixture's model - the same
  // whatever QMD_EMBED_MODEL the worker's environment carries.
  override get embedModelName(): string {
    return LEGACY_ADOPTION_STUB_MODEL;
  }

  override async tokenize(text: string): Promise<Token[]> {
    return new Array<Token>(Math.max(1, Math.ceil(text.length / 16))).fill(stubToken);
  }

  override async detokenize(tokens: readonly Token[]): Promise<string> {
    return "x".repeat(tokens.length * 16);
  }

  override async embed(): Promise<EmbeddingResult> {
    return { embedding: [0.1, 0.2, 0.3], model: LEGACY_ADOPTION_STUB_MODEL };
  }
}
