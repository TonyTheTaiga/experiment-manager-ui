import { generateText } from "$lib/server/llm";
import { getExperimentAndMetrics } from "$lib/server/database";
import type { ExperimentAndMetrics } from "$lib/types";
import { formatExperimentForLLM } from "$lib/server/analysis/prompts";

export async function GET({ params: { slug } }: { params: { slug: string } }) {
  const experiment = (await getExperimentAndMetrics(
    slug,
  )) as ExperimentAndMetrics;

  const prompt = formatExperimentForLLM(experiment);
  console.log(prompt);
  console.log(slug)

  const analysis = await generateText(prompt);
  return new Response(JSON.stringify({ analysis }));
}
