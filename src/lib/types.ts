export type Experiment = {
  id: number;
  name: string;
  description: string | null;
  groups?: string[];
  availableMetrics?: string[];
  hyperparams?: HyperParam[];
  createdAt?: Date;
};

export type HyperParam = {
  key: string;
  value: any;
};
