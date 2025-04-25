import { PUBLIC_ANTHROPIC_KEY } from "$env/static/public";
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: PUBLIC_ANTHROPIC_KEY,
});

const MODEL = "claude-3-7-sonnet-20250219";

const BASE_PROMPT =
  "You are a helpful assistant tasked to analyze and compile a report of machine learning experiments.";

const TEXT_PROMPT = `${BASE_PROMPT} Output well structured text-only markdown.`;

export const generateText = async (prompt: string) => {
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    system: TEXT_PROMPT,
  });
  console.log(msg);
  if (msg.content[0].type === "text") {
    return msg.content[0].text;
  }
  return "Error: No text response received";
};

const STRUCTURED_PROMPT = `${BASE_PROMPT} 
Output a structured JSON analysis of the experiment with the following fields:
- summary: A brief overview of the experiment (1-2 sentences)
- performance: Analysis of key metrics and their trends
- insights: Key takeaways and patterns observed
- recommendations: Suggested next steps or improvements

Your analysis must be valid JSON with these exact keys. Do not include any text outside the JSON structure.`;

export const generateJSON = async (prompt: string) => {
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
    system: STRUCTURED_PROMPT,
    tools: [
      {
        name: "analysis_schema",
        description: "Schema for experiment analysis output",
        input_schema: {
          type: "object",
          properties: {
            summary: {
              type: "string",
              description: "A brief overview of the experiment (1-2 sentences)",
            },
            performance: {
              type: "object",
              description: "Analysis of key metrics and their trends",
              properties: {
                overall: {
                  type: "string",
                  description: "Overall assessment of experiment performance",
                },
                metrics: {
                  type: "object",
                  description: "Analysis of individual metrics",
                  additionalProperties: {
                    type: "string",
                    description: "Analysis of a specific metric",
                  },
                },
              },
              required: ["overall"],
            },
            insights: {
              type: "array",
              description: "Key takeaways and patterns observed",
              items: {
                type: "string",
              },
            },
            recommendations: {
              type: "array",
              description: "Suggested next steps or improvements",
              items: {
                type: "string",
              },
            },
          },
          required: ["summary", "performance", "insights", "recommendations"],
        },
      },
    ],
    tool_choice: {
      type: "tool",
      name: "analysis_schema",
    },
  });
  console.log(msg);

  // Extract tool response if available, otherwise fall back to text content
  if (
    msg.content[0].type === "tool_use" &&
    msg.content[0].name === "analysis_schema"
  ) {
    return JSON.stringify(msg.content[0].input);
  }

  if (msg.content[0].type === "text") {
    return msg.content[0].text;
  }

  return JSON.stringify({ error: "Unable to generate structured analysis" });
};
