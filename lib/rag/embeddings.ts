import { pipeline } from '@xenova/transformers';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let embeddingPipeline: any = null;
let initPromise: Promise<void> | null = null;

const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';
export const EMBEDDING_DIMENSION = 384;

// Init and generate embeddings - runs on the server, no API call
// Singleton pattern - model loads once and is reused across requests
async function initEmbeddingModel(): Promise<void> {
  if (embeddingPipeline) return;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    console.log('Initializing embedding model (first run downloads ~23MB)...');
    embeddingPipeline = await pipeline('feature-extraction', MODEL_NAME, {
      quantized: true,
    });
    console.log('Embedding model ready');
  })();

  return initPromise;
}

export async function generateEmbedding(text: string): Promise<number[]> {
  await initEmbeddingModel();

  const output = await embeddingPipeline!(text, {
    pooling: 'mean',
    normalize: true,
  });
  // Return 384-dimensional vector of the text
  return Array.from(output.data as Float32Array);
}