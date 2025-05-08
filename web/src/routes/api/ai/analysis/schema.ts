import { z } from 'zod';

export const HPRecommendationSchema = z.object({
  recommendation: z.string().max(32),
  importance_level: z.enum(['1', '2', '3', '4', '5']).transform((s) => parseInt(s) as 1 | 2 | 3 | 4 | 5),
});

export type HPRecommendation = z.infer<typeof HPRecommendationSchema>;

export const AnalysisSchema = z.object({
  summary: z.string(),
  hyperparameter_recommendations: z.record(HPRecommendationSchema),
});

export type OutputSchemaType = z.infer<typeof AnalysisSchema>;

// Usage
// import { zodToJsonSchema } from 'zod-to-json-schema';
// const jsonSchema = zodToJsonSchema(AnalysisSchema, "OutputSchema");
// console.log(JSON.stringify(jsonSchema, null, 2));
