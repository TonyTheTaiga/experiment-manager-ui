export type Experiment = {
    id: number,
    name: string;
    groups?: string[];
    availableMetrics?: string[];
    running: boolean;
}