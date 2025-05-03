import { PUBLIC_ANTHROPIC_KEY } from "$env/static/public";
import Anthropic from "@anthropic-ai/sdk";

export function createAnthropicClient() {
  const client = new Anthropic({
    apiKey: PUBLIC_ANTHROPIC_KEY,
  });
  return client;
}