import { json } from "@sveltejs/kit";
import { createMetric } from "$lib/server/database.js";
import type { Json } from "$lib/server/database.types.js";

export async function POST({ request, params }) {
  const experimentId = params.slug;
  const {
    name,
    value,
    metadata,
    step,
  }: { name: string; value: number; metadata?: Json; step?: number } =
    await request.json();
  await createMetric(experimentId, name, value, step, metadata);
  return json({ success: true });
}
