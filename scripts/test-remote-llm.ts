#!/usr/bin/env bun
import { loadLLMConfig, getLLMConfigPath, type OpenAIConfig } from "../src/llm_config";

type OpenAIModelsResponse = {
  data?: { id: string }[];
};

type EmbeddingResponse = {
  data: { embedding: number[]; index?: number }[];
};

type ChatResponse = {
  choices?: { message?: { content?: string } }[];
};

type ResponsesOutput = {
  output?: { content?: { type?: string; text?: string }[] }[];
  output_text?: string;
};

function resolveBaseUrl(config: OpenAIConfig): string {
  if (config.base_url) {
    return config.base_url.replace(/\/+$/, "");
  }
  const protocol = config.protocol || "http";
  const host = config.host || "localhost";
  const port = config.port ?? 8000;
  return `${protocol}://${host}:${port}`;
}

function buildHeaders(apiKey?: string): HeadersInit {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

function extractResponsesOutputText(payload: ResponsesOutput): string | null {
  if (typeof payload.output_text === "string") {
    return payload.output_text;
  }
  if (Array.isArray(payload.output)) {
    for (const item of payload.output) {
      const content = item?.content;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (part?.type === "output_text" && typeof part.text === "string") {
          return part.text;
        }
      }
    }
  }
  return null;
}

async function main(): Promise<void> {
  const config = loadLLMConfig();
  if (config.provider !== "openai" || !config.openai) {
    console.error(`OpenAI provider is not configured. Edit ${getLLMConfigPath()}.`);
    process.exit(1);
  }

  const openai = config.openai;
  const baseUrl = resolveBaseUrl(openai);
  const headers = buildHeaders(openai.api_key);

  console.log(`Using base URL: ${baseUrl}`);

  const modelsResp = await fetch(`${baseUrl}/v1/models`, { headers });
  if (!modelsResp.ok) {
    throw new Error(`Model list request failed: ${modelsResp.status} ${await modelsResp.text()}`);
  }
  const models = await modelsResp.json() as OpenAIModelsResponse;
  const modelIds = models.data?.map((model) => model.id) ?? [];
  console.log(`Models (${modelIds.length}): ${modelIds.slice(0, 10).join(", ")}`);

  if (!openai.models?.embed || !openai.models?.generate || !openai.models?.rerank) {
    throw new Error(`Missing model identifiers in ${getLLMConfigPath()}.`);
  }

  const embedResp = await fetch(`${baseUrl}/v1/embeddings`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: openai.models.embed,
      input: "test embedding from qmd",
    }),
  });
  if (!embedResp.ok) {
    throw new Error(`Embeddings request failed: ${embedResp.status} ${await embedResp.text()}`);
  }
  const embedData = await embedResp.json() as EmbeddingResponse;
  const embeddingLength = embedData.data?.[0]?.embedding?.length ?? 0;
  console.log(`Embedding length: ${embeddingLength}`);

  const chatResp = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: openai.models.generate,
      temperature: openai.temperatures?.generate ?? 0.7,
      max_tokens: 120,
      messages: [
        {
          role: "user",
          content: "Return a single short sentence about vector search.",
        },
      ],
    }),
  });
  if (!chatResp.ok) {
    throw new Error(`Chat request failed: ${chatResp.status} ${await chatResp.text()}`);
  }
  const chatData = await chatResp.json() as ChatResponse;
  const chatText = chatData.choices?.[0]?.message?.content?.trim() ?? "";
  console.log(`Chat response: ${chatText || "(empty)"}`);

  const rerankPrompt =
    "Score each document for relevance to the query on a 0-1 scale.\n" +
    "Return a JSON array of objects with fields: index, score.\n" +
    "Query: vector search\n" +
    "Document 0:\nVector search uses embeddings to find similar text.\n\n" +
    "Document 1:\nThis document is about cooking pasta.";

  if (openai.responses?.rerank) {
    const rerankResp = await fetch(`${baseUrl}/v1/responses`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: openai.models.rerank,
        input: rerankPrompt,
        temperature: openai.temperatures?.rerank ?? 0.1,
        max_output_tokens: 200,
        response_format: {
          type: "json_schema",
          json_schema: {
            type: "array",
            items: {
              type: "object",
              additionalProperties: false,
              properties: {
                index: { type: "integer" },
                score: { type: "number" },
              },
              required: ["index", "score"],
            },
          },
        },
      }),
    });
    if (!rerankResp.ok) {
      throw new Error(`Rerank request failed: ${rerankResp.status} ${await rerankResp.text()}`);
    }
    const rerankData = await rerankResp.json() as ResponsesOutput;
    const rerankText = extractResponsesOutputText(rerankData)?.trim() ?? "";
    console.log(`Rerank response: ${rerankText || "(empty)"}`);
  } else {
    const rerankResp = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model: openai.models.rerank,
        temperature: openai.temperatures?.rerank ?? 0.1,
        max_tokens: 200,
        messages: [
          {
            role: "user",
            content: rerankPrompt,
          },
        ],
      }),
    });
    if (!rerankResp.ok) {
      throw new Error(`Rerank request failed: ${rerankResp.status} ${await rerankResp.text()}`);
    }
    const rerankData = await rerankResp.json() as ChatResponse;
    const rerankText = rerankData.choices?.[0]?.message?.content?.trim() ?? "";
    console.log(`Rerank response: ${rerankText || "(empty)"}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
