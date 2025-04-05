import type { Json } from "./server/database.types";

export interface ExperimentAndMetrics {
  experiment: Experiment;
  metrics: Metric[];
}

export interface Experiment {
  id: string;
  name: string;
  description?: string | null;
  availableMetrics?: string[] | null;
  hyperparams?: HyperParam[] | null;
  tags?: string[] | null;
  createdAt: Date;
}

export interface Metric {
  id: number;
  experiment_id: string;
  name: string;
  value: number;
  step?: number;
  metadata?: Json;
  created_at: string;
}

export interface HyperParam {
  key: string;
  value: string | number;
}
