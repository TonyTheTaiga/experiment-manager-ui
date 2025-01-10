import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";
import {
  PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY,
} from "$env/static/public";
import type { Database, Json } from "./database.types";
import { type Experiment, type HyperParam } from "$lib/types";
const supabaseUrl = PUBLIC_SUPABASE_URL;
const supabaseKey = PUBLIC_SUPABASE_ANON_KEY;

let client: SupabaseClient<Database>;

function getClient() {
  if (!client) {
    client = createClient<Database>(supabaseUrl, supabaseKey);
  }

  return client;
}

export async function createExperiment(
  name: string,
  description: string,
  hyperparams: HyperParam[],
): Promise<Experiment> {
  client = getClient();
  const { data, error } = await client
    .from("experiment")
    .insert({ name: name, description: description, hyperparams: hyperparams })
    .select();
  if (error) {
    throw new Error("Failed to create experiment");
  }
  let createdAt = new Date(data[0].created_at);
  return {
    name: data[0].name,
    description: data[0].description,
    createdAt: createdAt,
    id: data[0].id,
    hyperparams: data[0].hyperparams as HyperParam[],
  };
}

export async function getExperiments(): Promise<Experiment[]> {
  client = getClient();
  const { data, error } = await client.from("experiment").select();
  if (error) {
    throw new Error("Failed to fetch experiments");
  }

  let experiments = data.map((query_data) => ({
    id: query_data["id"],
    name: query_data["name"],
    description: query_data["description"],
    hyperparams: query_data["hyperparams"] as HyperParam[],
    createdAt: new Date(query_data["created_at"]),
  }));

  return experiments;
}

export async function getExperiment(id: number) {
  client = getClient();
  return await client.from("experiment").select().eq("id", id);
}

export async function deleteExeriment(id: number) {
  client = getClient();
  await client.from("experiment").delete().eq("id", id);
}

export async function createMetric(
  experimentId: string,
  name: string,
  value: number,
  step?: number,
  metadata?: Json,
) {
  client = getClient();
  const { error } = await client.from("metric").insert({
    experiment_id: experimentId,
    name: name,
    value: value,
    metadata: metadata,
    step: step,
  });
  if (error) {
    throw new Error("Failed to write metric");
  }
}
