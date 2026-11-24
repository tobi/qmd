/**
 * Query-expansion contract for the default generate model (v2-VH SFT).
 *
 * These values are the "profile D" serving configuration the model was
 * evaluated under; the online expansion path must use them verbatim. Do not
 * edit them without re-running the retrieval evaluation — the prompt is the
 * exact training prompt, pinned by VH_PROMPT_SHA256.
 */

export const VH_PROMPT_VERSION = "qmd-student-expansion-v2-vh-v1";

/** Full prompt template; {query} is the placeholder. */
export const VH_PROMPT_TEMPLATE = "/no_think You generate retrieval expansions for a search query.\n\nReturn only newline-separated lines in exactly this format:\nhyde: <one hypothetical document passage>\nvec: <semantic search query>\nvec: <optional second complementary semantic search query>\n\nRules:\n- Return exactly one hyde line, followed by one or two vec lines.\n- Use only the prefixes `hyde:` and `vec:`.\n- Preserve the query's entities, constraints, negation, comparison, and intent.\n- Make Vec queries complementary rather than duplicates.\n- Write a faithful hypothetical passage for HyDE; do not answer the user or invent unsupported facts.\n\nSearch query: {query}";

export const VH_PROMPT_SHA256 = "fce6b582fff10e7e1a601097e0fa00273f5b106fcd821985422932a5ce797d93";

/** One hyde line, then one or two vec lines; the final newline is optional. */
export const VH_GRAMMAR = "root ::= hydeLine vecLine vecLine?\nhydeLine ::= \"hyde: \" [^\\n\\r]+ \"\\n\"\nvecLine ::= \"vec: \" [^\\n\\r]+ (\"\\n\")?\n";

/** Empty system turn, matching the SFT training envelope. */
export const VH_SYSTEM_PROMPT = "";

/** Greedy decoding; lastTokens 0 disables the repeat/presence penalty. */
export const VH_SAMPLING = {
  temperature: 0,
  topK: 20,
  topP: 0.8,
  presencePenalty: 0,
  lastTokens: 0,
} as const;

export const VH_MAX_TOKENS = 600;
