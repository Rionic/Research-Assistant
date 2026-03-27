export interface Chunk {
  text: string;
  index: number;
}

const MAX_CHUNK_SIZE = 800;
const CHUNK_OVERLAP = 100;

// Straightforward. Chunks the text based on the defined chunk size & overlap
export function chunkText(text: string): Chunk[] {
  if (!text || text.trim().length === 0) {
    return [];
  }

  const paragraphs = text.split(/\n\n+/);
  const chunks: Chunk[] = [];
  let currentChunk = '';
  let chunkIndex = 0;

  for (const para of paragraphs) {
    const trimmedPara = para.trim();
    if (!trimmedPara) continue;
    
    if (currentChunk.length + trimmedPara.length > MAX_CHUNK_SIZE && currentChunk) {
      chunks.push({
        text: currentChunk.trim(),
        index: chunkIndex++,
      });

      const overlapText = currentChunk.slice(-CHUNK_OVERLAP).trim();
      currentChunk = overlapText + ' ' + trimmedPara;
    } else {
      currentChunk += (currentChunk ? '\n\n' : '') + trimmedPara;
    }
  }

  if (currentChunk.trim()) {
    chunks.push({
      text: currentChunk.trim(),
      index: chunkIndex,
    });
  }

  return chunks;
}
