import { generateJSON } from "$lib/server/llm";
import { getExperimentAndMetrics } from "$lib/server/database";
import type { ExperimentAndMetrics } from "$lib/types";

/**
 * Structured analysis API endpoint
 * Returns JSON-structured analysis of experiment data
 */
export async function GET({ url }: { url: URL }) {
  const experimentId = url.searchParams.get("experimentId");

  if (!experimentId) {
    return new Response(JSON.stringify({ error: "experimentId is required" }), {
      status: 400,
    });
  }

  const experiment = (await getExperimentAndMetrics(
    experimentId,
  )) as ExperimentAndMetrics;

  // Format experiment data for LLM
  const formattedData = formatExperimentStructured(experiment);

  // Generate JSON analysis
  const analysis = await generateJSON(formattedData);

  // Parse the JSON response to return as actual JSON
  try {
    const parsedAnalysis = JSON.parse(analysis);
    return new Response(JSON.stringify(parsedAnalysis), {
      headers: {
        "Content-Type": "application/json",
      },
    });
  } catch (error) {
    console.error("Failed to parse LLM response as JSON:", error);
    return new Response(
      JSON.stringify({
        error: "Failed to generate structured analysis",
        raw: analysis,
      }),
      { status: 500 },
    );
  }
}

/**
 * Formats experiment data specifically for structured analysis
 */
function formatExperimentStructured(data: ExperimentAndMetrics): string {
  const { experiment, metrics } = data;

  // Begin with experiment overview
  let formatted = `# Experiment Analysis Request\n\n`;
  formatted += `Analyze experiment: ${experiment.name} (ID: ${experiment.id})\n\n`;

  // Add description if available
  if (experiment.description) {
    formatted += `## Description\n${experiment.description}\n\n`;
  }

  // Add tags
  if (experiment.tags && experiment.tags.length > 0) {
    formatted += `## Tags\n${experiment.tags.join(", ")}\n\n`;
  }

  // Add hyperparameters
  if (experiment.hyperparams && experiment.hyperparams.length > 0) {
    formatted += `## Hyperparameters\n`;
    experiment.hyperparams.forEach((param) => {
      formatted += `- ${param.key}: ${param.value}\n`;
    });
    formatted += "\n";
  }

  // Add metrics summary
  formatted += `## Metrics Summary\n`;

  if (metrics.length === 0) {
    formatted += "No metrics recorded.\n\n";
  } else {
    // Group metrics by name
    const metricsByName = metrics.reduce(
      (acc, m) => {
        if (!acc[m.name]) acc[m.name] = [];
        acc[m.name].push(m);
        return acc;
      },
      {} as Record<string, typeof metrics>,
    );

    // Add summary stats for each metric
    Object.entries(metricsByName).forEach(([name, values]) => {
      const numericValues = values.map((m) => m.value);
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const avg =
        numericValues.reduce((sum, val) => sum + val, 0) / numericValues.length;
      const latest = values[values.length - 1].value;

      formatted += `### ${name}\n`;
      formatted += `- Latest: ${latest}\n`;
      formatted += `- Min: ${min}\n`;
      formatted += `- Max: ${max}\n`;
      formatted += `- Avg: ${avg.toFixed(4)}\n`;
      formatted += `- Count: ${values.length}\n\n`;
    });
  }

  // Add analysis guidance
  formatted += `## Analysis Request\n`;
  formatted += `Please analyze this experiment data and provide insights on performance, trends, and potential improvements.\n`;
  formatted += `Focus on providing specific, actionable recommendations based on the metrics and parameters.\n`;

  return formatted;
}

