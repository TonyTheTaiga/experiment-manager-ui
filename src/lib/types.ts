import type { Json } from "./server/database.types";

export interface Experiment {
  id: string;
  name: string;
  description?: string | null;
  availableMetrics?: string[] | null;
  hyperparams?: HyperParam[] | null;
  createdAt: Date;
};

export interface Metric {
  experiment_id: string,
  name: string,
  value: number,
  step?: number,
  metadata?: Json,
}

export interface HyperParam {
  key: string;
  value: string | number;
};
