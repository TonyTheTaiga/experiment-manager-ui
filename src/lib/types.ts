export type Experiment = {
  id: string;
  name: string;
  description?: string | null;
  groups?: string[];
  availableMetrics?: string[] | null;
  hyperparams?: HyperParam[] | null;
  createdAt?: Date;
};

export type HyperParam = {
  key: string;
  value: string | number;
};
