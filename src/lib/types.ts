export type Experiment = {
    id: number,
    name: string;
    groups?: string[];
    running: boolean;
}