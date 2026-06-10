/**
 * vitest setup: when QMD_REMOTE_URL is set, register a RemoteLLM as the
 * default so integration tests share `qmd serve`'s resident models instead
 * of spawning their own LlamaCpp on the GPU (which collides with the running
 * serve and fails "context size too large for the available VRAM").
 */
import { setDefaultLLM } from "./llm.js";
import { RemoteLLM } from "./llm-remote.js";

const remoteUrl = process.env.QMD_REMOTE_URL;
if (remoteUrl) {
  setDefaultLLM(new RemoteLLM({ serverUrl: remoteUrl }));
}
