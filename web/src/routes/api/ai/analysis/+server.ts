import { getExperimentAndMetrics } from '$lib/server/database';
import type { ExperimentAndMetrics } from '$lib/types';
import { createAnthropicClient } from '$lib/server/llm';

const MODEL = 'claude-3-7-sonnet-20250219';

export async function GET({ url }: { url: URL }) {
  const experimentId = url.searchParams.get('experimentId');
  if (!experimentId) {
    return new Response(JSON.stringify({ error: 'experimentId is required' }), { status: 400 });
  }

  const data = (await getExperimentAndMetrics(experimentId)) as ExperimentAndMetrics;
  const system = createSystemPrompt();
  const user = createUserPrompt(data);

  try {
    const client = createAnthropicClient();
    const msg = await client.messages.create({
      model: MODEL,
      system,
      messages: [{ role: 'user', content: user }],
      max_tokens: 1024,
    });

    const raw = msg.content[0].type === 'text'
      ? msg.content[0].text
      : JSON.stringify({ error: 'invalid model response' });


    console.log(raw);

    const parsed = parseOutput(raw);
    return new Response(JSON.stringify(parsed), { headers: { 'Content-Type': 'application/json' } });
  } catch (err) {
    console.error('LLM analysis error', err);
    return new Response(JSON.stringify({ error: 'structured analysis failed' }), { status: 500 });
  }
}

function createSystemPrompt(): string {
  const BASE = 'You are a helpful assistant tasked to analyze and compile a report of machine learning experiments.';
  return `${BASE}\nOutput a structured JSON analysis with keys: summary, performance, insights, recommendations. Only JSON.`;
}

function createUserPrompt({ experiment, metrics }: ExperimentAndMetrics): string {
  const lines: string[] = [
    '# Experiment Analysis Request',
    `Analyze experiment: ${experiment.name} (ID: ${experiment.id})`,
  ];

  if (experiment.description) {
    lines.push('## Description', experiment.description);
  }
  if (experiment.tags?.length) {
    lines.push('## Tags', experiment.tags.join(', '));
  }
  if (experiment.hyperparams?.length) {
    lines.push('## Hyperparameters');
    experiment.hyperparams.forEach(({ key, value }) => {
      lines.push(`- ${key}: ${value}`);
    });
  }

  lines.push('## Metrics Summary');
  if (!metrics.length) {
    lines.push('No metrics recorded.');
  } else {
    const grouped = metrics.reduce<Record<string, typeof metrics>>((acc, m) => {
      (acc[m.name] ||= []).push(m);
      return acc;
    }, {});

    for (const [name, entries] of Object.entries(grouped)) {
      const values = entries.map(e => e.value);
      const latest = values.at(-1) ?? 0;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const avg = values.reduce((sum, v) => sum + v, 0) / values.length;

      lines.push(
        `### ${name}`,
        `- Latest: ${latest}`,
        `- Min: ${min}`,
        `- Max: ${max}`,
        `- Avg: ${avg.toFixed(4)}`,
        `- Count: ${values.length}`
      );
    }
  }

  lines.push(
    '## Analysis Request',
    'Please analyze this experiment data and provide insights on performance, trends, and improvements.'
  );

  return lines.join('\n\n');
}

function parseOutput(raw: string) {
  // Remove markdown code fences if present
  let jsonText = raw.trim();
  if (jsonText.startsWith('```json')) {
    jsonText = jsonText.slice(7).trim();
  } else if (jsonText.startsWith('```')) {
    jsonText = jsonText.slice(3).trim();
  }
  if (jsonText.endsWith('```')) {
    jsonText = jsonText.slice(0, -3).trim();
  }

  try {
    return JSON.parse(jsonText);
  } catch (e) {
    console.warn(`Failed to parse JSON output: ${e}`);
    throw e;
  }
}

