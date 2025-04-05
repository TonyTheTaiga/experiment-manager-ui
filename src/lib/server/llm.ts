import { PUBLIC_ANTHROPIC_KEY } from "$env/static/public";
import Anthropic from "@anthropic-ai/sdk";
import type { Tool } from "@anthropic-ai/sdk/resources/index";

const client = new Anthropic({
  apiKey: PUBLIC_ANTHROPIC_KEY,
});

const MODEL = "claude-3-7-sonnet-20250219";

const BASE_PROMPT =
  "You are a helpful assistant tasked to analyze and compile a report of machine learning experiments.";

const TEXT_PROMPT = `${BASE_PROMPT} Output well structured text-only markdown.`;

const STRUCTURED_PROMPT = `${BASE_PROMPT} Output well structured JSON.`;

export const generateText = async (prompt: string) => {
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    system: TEXT_PROMPT,
  });
  console.log(msg);
  return msg.content[0].text;
};

export const generateJSON = async (prompt: string) => {
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    system: STRUCTURED_PROMPT,
  });
  console.log(msg);
  return msg.content[0].text;
};
