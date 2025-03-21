import { PUBLIC_ANTHROPIC_KEY } from "$env/static/public";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: PUBLIC_ANTHROPIC_KEY,
});

const PROMPT =
  "You are a helpful assistant tasked to analyze and compile a report of machine learning experiments";

export const generateText = async (prompt: string) => {
  const msg = await client.messages.create({
    model: "claude-3-7-sonnet-20250219",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    system: PROMPT,
  });
  console.log(msg);
  return msg.content[0].text;
};
