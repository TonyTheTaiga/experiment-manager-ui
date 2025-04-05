import { json, type RequestEvent } from "@sveltejs/kit";
import { batchCreateMetric } from "$lib/server/database";
import type { Json } from "$lib/server/database.types";

interface MetricInput {
  name: string;
  value: number;
  step?: number;
  metadata?: Json;
}

interface APIError {
  message: string;
  code: string;
  status: number;
}

export async function POST({
  request,
  params,
}: RequestEvent<{ slug: string }, string>): Promise<Response> {
  try {
    const metrics = (await request.json()) as MetricInput[];
    if (!Array.isArray(metrics)) {
      throw new Error("Invalid input: expected array of metrics");
    }

    if (!metrics.every(isValidMetric)) {
      throw new Error("Invalid metric format");
    }

    const experimentId = params.slug;
    if (!experimentId?.trim()) {
      throw new Error("Invalid experiment ID");
    }

    const finalMetrics = metrics.map((data) => ({
      ...data,
      experiment_id: experimentId,
    }));

    await batchCreateMetric(finalMetrics);

    return json({
      success: true,
      count: metrics.length,
      experimentId,
    });
  } catch (error) {
    const apiError: APIError = {
      message:
        error instanceof Error ? error.message : "Unknown error occurred",
      code: "METRIC_CREATE_ERROR",
      status: 400,
    };

    return json(apiError, { status: apiError.status });
  }
}

function isValidMetric(metric: unknown): metric is MetricInput {
  if (!metric || typeof metric !== "object") return false;

  const m = metric as MetricInput;

  return (
    typeof m.name === "string" &&
    m.name.trim().length > 0 &&
    typeof m.value === "number" &&
    (m.step === undefined || typeof m.step === "number") &&
    (m.metadata === undefined || typeof m.metadata === "object")
  );
}
