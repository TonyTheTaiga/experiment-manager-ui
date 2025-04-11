import { json, type RequestEvent } from "@sveltejs/kit";
import { createMetric, getMetrics } from "$lib/server/database";
import type { Json } from "$lib/server/database.types";
import { MoveVerticalIcon } from "lucide-svelte";

interface MetricInput {
  name: string;
  value: number;
  metadata?: Json;
  step?: number;
}

interface APIResponse {
  success: boolean;
  error?: {
    message: string;
    code: string;
  };
}

export async function POST({
  request,
  params,
}: RequestEvent<{ slug: string }, string>): Promise<Response> {
  try {
    const payload = (await request.json()) as MetricInput;

    if (!payload.name?.trim()) {
      throw new Error("Metric name is required");
    }

    if (payload.value === undefined || payload.value === null) {
      throw new Error("Metric value is required");
    }

    const experimentId = params.slug;
    if (!experimentId?.trim()) {
      throw new Error("Invalid experiment ID");
    }

    if (payload.step !== undefined && !Number.isFinite(payload.step)) {
      throw new Error("Step must be a finite number");
    }

    await createMetric({
      experiment_id: experimentId,
      name: payload.name,
      value: payload.value,
      step: payload.step,
      metadata: payload.metadata,
    });

    return json({
      success: true,
    } satisfies APIResponse);
  } catch (error) {
    const statusCode = error instanceof Error ? 400 : 500;

    return json(
      {
        success: false,
        error: {
          message:
            error instanceof Error ? error.message : "Internal server error",
          code: "METRIC_CREATE_FAILED",
        },
      } satisfies APIResponse,
      { status: statusCode },
    );
  }
}

export async function GET({ params: { slug } }: { params: { slug: string } }) {
  const metrics = await getMetrics(slug);
  return json(metrics);
}
