import { randomUUID } from 'crypto';
import { ResearchSession } from '@/types';
import { generateEmbedding } from './embeddings';
import { chunkText } from './chunker';
import {
  ensureCollection,
  upsertVectors,
  searchSimilar,
  VectorPoint,
  SearchResult,
} from './qdrant';

export interface RAGContext {
  relevantResults: SearchResult[];
}

const TOP_K_DEFAULT = 5;
const SIMILARITY_THRESHOLD = 0.6;

export async function embedResearchResults(session: ResearchSession): Promise<void> {
  try {
    await ensureCollection();

    const points: VectorPoint[] = [];

    if (session.openaiResult) {
      const chunks = chunkText(session.openaiResult);
      for (const chunk of chunks) {
        const embedding = await generateEmbedding(chunk.text);
        points.push({
          id: randomUUID(),
          vector: embedding,
          payload: {
            sessionId: session.id,
            userId: session.userId,
            initialPrompt: session.initialPrompt,
            refinedPrompt: session.refinedPrompt,
            source: 'openai',
            chunkIndex: chunk.index,
            chunkText: chunk.text,
            createdAt: new Date().toISOString(),
          },
        });
      }
    }

    if (session.geminiResult) {
      const chunks = chunkText(session.geminiResult);
      for (const chunk of chunks) {
        const embedding = await generateEmbedding(chunk.text);
        points.push({
          id: randomUUID(),
          vector: embedding,
          payload: {
            sessionId: session.id,
            userId: session.userId,
            initialPrompt: session.initialPrompt,
            refinedPrompt: session.refinedPrompt,
            source: 'gemini',
            chunkIndex: chunk.index,
            chunkText: chunk.text,
            createdAt: new Date().toISOString(),
          },
        });
      }
    }

    if (points.length > 0) {
      await upsertVectors(points);
      console.log(`Embedded ${points.length} chunks for session ${session.id}`);
    }
  } catch (error) {
    console.error('Failed to embed research results:', error);
  }
}

export async function retrieveContext(
  prompt: string,
  userId: string,
  topK: number = TOP_K_DEFAULT
): Promise<RAGContext> {
  try {
    await ensureCollection();
    const queryEmbedding = await generateEmbedding(prompt);
    const results = await searchSimilar(queryEmbedding, userId, topK);
    const filteredResults = results.filter(r => r.similarity >= SIMILARITY_THRESHOLD);

    return { relevantResults: filteredResults };
  } catch (error) {
    console.error('Failed to retrieve RAG context:', error);
    return { relevantResults: [] };
  }
}

export function augmentPrompt(originalPrompt: string, context: RAGContext): string {
  if (context.relevantResults.length === 0) {
    return originalPrompt;
  }

  const contextSection = context.relevantResults
    .map(
      (r, i) =>
        `[Previous Research ${i + 1}] (Similarity: ${(r.similarity * 100).toFixed(1)}%)\nOriginal Query: ${r.originalPrompt}\nRelevant Finding: ${r.text}`
    )
    .join('\n\n');

  return `${originalPrompt}

---
The following are relevant findings from previous research that may provide helpful context:

${contextSection}

---
Please incorporate any relevant insights from the above context into your research, while focusing primarily on the main research query.`;
}
