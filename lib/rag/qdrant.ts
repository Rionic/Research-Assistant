import { QdrantClient } from '@qdrant/js-client-rest';
import { EMBEDDING_DIMENSION } from './embeddings';

export const COLLECTION_NAME = 'research_results';

let qdrantClient: QdrantClient | null = null;

export function getQdrantClient(): QdrantClient {
  if (!qdrantClient) {
    const url = process.env.QDRANT_URL;
    const apiKey = process.env.QDRANT_API_KEY;

    if (!url || !apiKey) {
      throw new Error('QDRANT_URL and QDRANT_API_KEY must be set');
    }

    qdrantClient = new QdrantClient({
      url,
      apiKey,
    });
  }

  return qdrantClient;
}

export async function ensureCollection(): Promise<void> {
  const client = getQdrantClient();

  try {
    const collections = await client.getCollections();
    const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

    if (!exists) {
      console.log(`Creating Qdrant collection: ${COLLECTION_NAME}`);

      await client.createCollection(COLLECTION_NAME, {
        vectors: {
          size: EMBEDDING_DIMENSION,
          distance: 'Cosine',
        },
      });

      await client.createPayloadIndex(COLLECTION_NAME, {
        field_name: 'userId',
        field_schema: 'keyword',
      });

      console.log(`Collection ${COLLECTION_NAME} created successfully`);
    }
  } catch (error) {
    console.error('Error ensuring Qdrant collection:', error);
    throw error;
  }
}

export interface VectorPoint {
  id: string;
  vector: number[];
  payload: {
    sessionId: string;
    userId: string;
    initialPrompt: string;
    refinedPrompt?: string;
    source: 'openai' | 'gemini';
    chunkIndex: number;
    chunkText: string;
    createdAt: string;
  };
}

export async function upsertVectors(points: VectorPoint[]): Promise<void> {
  if (points.length === 0) return;

  const client = getQdrantClient();

  await client.upsert(COLLECTION_NAME, {
    points: points.map(p => ({
      id: p.id,
      vector: p.vector,
      payload: p.payload,
    })),
  });
}

export interface SearchResult {
  text: string;
  similarity: number;
  source: 'openai' | 'gemini';
  originalPrompt: string;
  sessionId: string;
}

export async function searchSimilar(
  queryVector: number[],
  userId: string,
  topK: number = 5
): Promise<SearchResult[]> {
  const client = getQdrantClient();

  const results = await client.search(COLLECTION_NAME, {
    vector: queryVector,
    limit: topK,
    filter: {
      must: [
        {
          key: 'userId',
          match: { value: userId },
        },
      ],
    },
    with_payload: true,
  });

  return results.map(r => ({
    text: (r.payload?.chunkText as string) || '',
    similarity: r.score,
    source: (r.payload?.source as 'openai' | 'gemini') || 'openai',
    originalPrompt: (r.payload?.initialPrompt as string) || '',
    sessionId: (r.payload?.sessionId as string) || '',
  }));
}
