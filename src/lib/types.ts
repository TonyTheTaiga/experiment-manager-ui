export type Experiment = {
    id: number,
    name: string;
    description: string | null;
    groups?: string[];
    availableMetrics?: string[];
    jobState: number;
}