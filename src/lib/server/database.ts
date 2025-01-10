import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";
import {
  PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY,
} from "$env/static/public";
import type { Database, Json } from "./database.types";
import { type Experiment, type HyperParam, type Metric } from "$lib/types";
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
  let client = getClient();
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
  let client = getClient();

  const { data, error } = await client.from("experiment").select();
  if (error) {
    throw new Error("Failed to get experiments");
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
  let client = getClient();
  let { data, error } = await client.from("experiment").select().eq("id", id);
  if (error) {
    throw new Error("Failed to get experiment with ID: " + id);
  }

  return data;
}

export async function deleteExeriment(id: number) {
  let client = getClient();
  let { error } = await client.from("experiment").delete().eq("id", id);
  if (error) {
    throw new Error("Failed to get delete experiment with ID: " + id);
  }
}

export async function createMetric(metric: Metric) {
  let client = getClient();
  const { error } = await client.from("metric").insert(metric);
  if (error) {
    throw new Error("Failed to write metric");
  }
}

export async function batchCreateMetric(metrics: Metric[]) {
  let client = getClient();
  const maxRetries = 3;
  for (let i = 0; i < maxRetries; i++) {
    try {
      const { error } = await client.from("metric").insert(metrics);
      if (!error) return;

      throw error;
    } catch (error) {
      if (i === maxRetries - 1)
        throw new Error("Failed to write metrics after retries");
      await new Promise((resolve) => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}
