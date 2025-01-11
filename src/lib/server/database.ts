import { createClient } from "@supabase/supabase-js";
import type { SupabaseClient } from "@supabase/supabase-js";
import {
  PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY,
} from "$env/static/public";
import type { Database, Json } from "./database.types";
import type { Experiment, HyperParam, Metric } from "$lib/types";

class DatabaseClient {
  private static instance: SupabaseClient<Database>;

  private static getInstance(): SupabaseClient<Database> {
    if (!this.instance) {
      this.instance = createClient<Database>(
        PUBLIC_SUPABASE_URL,
        PUBLIC_SUPABASE_ANON_KEY,
      );
    }
    return this.instance;
  }

  static async createExperiment(
    name: string,
    description: string,
    hyperparams: HyperParam[],
    tags: string[],
  ): Promise<Experiment> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .insert({
        name,
        description,
        hyperparams: hyperparams as unknown as Json[],
        tags
      })
      .select()
      .single();

    if (error || !data) {
      throw new Error(`Failed to create experiment: ${error?.message}`);
    }

    return {
      id: data.id,
      name: data.name,
      description: data.description,
      hyperparams: data.hyperparams as unknown as HyperParam[],
      createdAt: new Date(data.created_at),
      tags: data.tags,
    };
  }

  static async getExperiments(): Promise<Experiment[]> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (name)")
      .order("created_at", { ascending: false });

    if (error) {
      throw new Error(`Failed to get experiments: ${error.message}`);
    }

    const result = data.map(
      (exp): Experiment => ({
        id: exp.id,
        name: exp.name,
        description: exp.description,
        hyperparams: exp.hyperparams as unknown as HyperParam[],
        createdAt: new Date(exp.created_at),
        tags: exp.tags,
        availableMetrics: [...new Set(exp.metric.map((m) => m.name))],
      }),
    );

    return result;
  }

  static async getExperiment(id: string): Promise<Experiment> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (name)")
      .eq("id", id)
      .single();

    if (error || !data) {
      throw new Error(
        `Failed to get experiment with ID ${id}: ${error?.message}`,
      );
    }

    return {
      id: data.id,
      name: data.name,
      description: data.description,
      hyperparams: data.hyperparams as unknown as HyperParam[],
      createdAt: new Date(data.created_at),
      availableMetrics: [...new Set(data.metric.map((m) => m.name))],
      tags: data.tags,
    };
  }

  static async deleteExperiment(id: string): Promise<void> {
    const { error } = await DatabaseClient.getInstance()
      .from("experiment")
      .delete()
      .eq("id", id);

    if (error) {
      throw new Error(
        `Failed to delete experiment with ID ${id}: ${error.message}`,
      );
    }
  }

  static async getMetrics(experimentId: string): Promise<Metric[]> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("metric")
      .select()
      .eq("experiment_id", experimentId)
      .order("created_at", { ascending: false });

    if (error) {
      throw new Error(`Failed to get metrics: ${error.message}`);
    }

    return data as Metric[];
  }

  static async createMetric(metric: Metric): Promise<void> {
    const { error } = await DatabaseClient.getInstance()
      .from("metric")
      .insert(metric);

    if (error) {
      throw new Error(`Failed to write metric: ${error.message}`);
    }
  }

  static async batchCreateMetric(metrics: Metric[]): Promise<void> {
    const maxRetries = 3;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const { error } = await DatabaseClient.getInstance()
          .from("metric")
          .insert(metrics);

        if (!error) return;
        lastError = new Error(`Batch insert failed: ${error.message}`);
      } catch (error) {
        lastError =
          error instanceof Error ? error : new Error("Unknown error occurred");

        if (attempt < maxRetries - 1) {
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * Math.pow(2, attempt)),
          );
          continue;
        }
      }
    }

    throw new Error(
      `Failed to write metrics after ${maxRetries} retries: ${lastError?.message}`,
    );
  }
}

export const {
  createExperiment,
  getExperiments,
  getExperiment,
  deleteExperiment,
  getMetrics,
  createMetric,
  batchCreateMetric,
} = DatabaseClient;
