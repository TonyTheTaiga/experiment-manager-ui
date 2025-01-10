import type { Json } from "./server/database.types";

export type Experiment = {
  id: string;
  name: string;
  description?: string | null;
  groups?: string[];
  availableMetrics?: string[] | null;
  hyperparams?: HyperParam[] | null;
  createdAt?: Date;
};

export type Metric = {
  experiment_id: string,
  name: string,
  value: number,
  step?: number,
  metadata?: Json,
}

export type HyperParam = {
  key: string;
  value: string | number;
};
