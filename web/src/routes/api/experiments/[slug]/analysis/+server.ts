import { generateText } from "$lib/server/llm";
import { getExperimentAndMetrics } from "$lib/server/database";
import type { ExperimentAndMetrics, Metric } from "$lib/types";


/**
 * Formats experiment and metrics data for LLM analysis, grouping metrics by name
 * @param data The experiment and metrics data
 * @returns A formatted string for LLM analysis
 */
function formatExperimentForLLM(data: ExperimentAndMetrics): string {
  const { experiment, metrics } = data;

  // Group metrics by name
  const metricsGroupedByName = metrics.reduce<Record<string, Metric[]>>(
    (acc, metric) => {
      if (!acc[metric.name]) {
        acc[metric.name] = [];
      }
      acc[metric.name].push(metric);
      return acc;
    },
    {},
  );

  // Format experiment details
  let formattedText = `# Experiment: ${experiment.name}\n\n`;
  formattedText += `ID: ${experiment.id}\n`;
  formattedText += `Created: ${experiment.createdAt.toISOString()}\n`;

  if (experiment.description) {
    formattedText += `\n## Description\n${experiment.description}\n`;
  }

  if (experiment.tags && experiment.tags.length > 0) {
    formattedText += `\n## Tags\n${experiment.tags.join(", ")}\n`;
  }

  if (experiment.hyperparams && experiment.hyperparams.length > 0) {
    formattedText += `\n## Hyperparameters\n`;
    experiment.hyperparams.forEach((param) => {
      formattedText += `- ${param.key}: ${param.value}\n`;
    });
  }

  if (experiment.availableMetrics && experiment.availableMetrics.length > 0) {
    formattedText += `\n## Available Metrics\n${experiment.availableMetrics.join(", ")}\n`;
  }

  // Format metrics by group
  formattedText += `\n## Metrics\n`;

  if (Object.keys(metricsGroupedByName).length === 0) {
    formattedText += "No metrics recorded for this experiment.\n";
  } else {
    Object.entries(metricsGroupedByName).forEach(
      ([metricName, metricsList]) => {
        formattedText += `\n### ${metricName}\n`;

        // Sort metrics by step if available, otherwise by creation date
        const sortedMetrics = [...metricsList].sort((a, b) => {
          if (a.step !== undefined && b.step !== undefined) {
            return a.step - b.step;
          }
          return (
            new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
          );
        });

        // Format metrics data based on whether they have steps
        const hasSteps = sortedMetrics.some((m) => m.step !== undefined);

        if (hasSteps) {
          formattedText += `| Step | Value | Timestamp |\n`;
          formattedText += `| ---- | ----- | --------- |\n`;
          sortedMetrics.forEach((metric) => {
            formattedText += `| ${metric.step !== undefined ? metric.step : "N/A"} | ${metric.value} | ${metric.created_at} |\n`;
          });
        } else {
          // If no steps, just list values
          formattedText += `Values: ${sortedMetrics.map((m) => m.value).join(", ")}\n`;
          formattedText += `Latest value: ${sortedMetrics[sortedMetrics.length - 1].value}\n`;
        }

        // Add metadata if available for any metric in this group
        const metricWithMetadata = sortedMetrics.find(
          (m) =>
            m.metadata !== undefined &&
            Object.keys(m.metadata || {}).length > 0,
        );
        if (metricWithMetadata?.metadata) {
          formattedText += `\nMetadata example:\n\`\`\`\n${JSON.stringify(metricWithMetadata.metadata, null, 2)}\n\`\`\`\n`;
        }
      },
    );
  }

  // Add summary statistics for each metric
  formattedText += `\n## Summary Statistics\n`;
  Object.entries(metricsGroupedByName).forEach(([metricName, metricsList]) => {
    const values = metricsList.map((m) => m.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;

    formattedText += `\n### ${metricName}\n`;
    formattedText += `- Count: ${values.length}\n`;
    formattedText += `- Min: ${min}\n`;
    formattedText += `- Max: ${max}\n`;
    formattedText += `- Average: ${avg.toFixed(4)}\n`;

    // Add trend analysis if multiple values
    if (values.length > 1) {
      const firstValue = values[0];
      const lastValue = values[values.length - 1];
      const change = lastValue - firstValue;
      const percentChange = (change / Math.abs(firstValue)) * 100;

      formattedText += `- Change: ${change > 0 ? "+" : ""}${change.toFixed(4)} (${percentChange > 0 ? "+" : ""}${percentChange.toFixed(2)}%)\n`;
      formattedText += `- Trend: ${change > 0 ? "Increasing" : change < 0 ? "Decreasing" : "Stable"}\n`;
    }
  });

  return formattedText;
}



export async function GET({ params: { slug } }: { params: { slug: string } }) {
  const experiment = (await getExperimentAndMetrics(
    slug,
  )) as ExperimentAndMetrics;

  const prompt = formatExperimentForLLM(experiment);
  const analysis = await generateText(prompt);
  return new Response(JSON.stringify({ analysis }));
}
