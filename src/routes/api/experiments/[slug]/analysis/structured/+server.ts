import { generateJSON, generateText } from "$lib/server/llm";
import { getExperimentAndMetrics } from "$lib/server/database";
import type { ExperimentAndMetrics } from "$lib/types";
import { formatExperimentForHyperparameterAnalysis } from "$lib/server/analysis/prompts";

function formatHyperparameterAnalysis(paramsNames: string[], output: string): string | null {
    try {
      const json = JSON.parse(output);
      for (const paramName of paramsNames) {
        if (!json[paramName]) {
          console.log("Failed on", paramName);
          return null;
        }
      }
      return json;
    } catch (error) {
      console.log("Failed on", error);
      return null;
    }
  }
  

export async function GET({ params: { slug } }: { params: { slug: string } }) {
    const experiment = (await getExperimentAndMetrics(
      slug,
    )) as ExperimentAndMetrics;

    if (!experiment.experiment.hyperparams) {
        return new Response(JSON.stringify({ error: "No hyperparameters found in this experiment" }), { status: 400 });
    }
  
    const prompt = formatExperimentForHyperparameterAnalysis(experiment);
  
    const analysis = await generateJSON(prompt);
    const formattedAnalysis = formatHyperparameterAnalysis(experiment.experiment.hyperparams.map(h => h.key), analysis);
    if (!formattedAnalysis) {
        return new Response(JSON.stringify({ error: "Invalid JSON output" }), { status: 400 });
    }
    return new Response(JSON.stringify(formattedAnalysis));
  }
  