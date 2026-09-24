export interface DocumentFile {
  id: string;
  name: string;
  size: string;
  category: string;
  chunksCount: number;
  status: 'Indexed' | 'Processing';
  lastUpdated: string;
}

export interface VectorChunk {
  id: string;
  chunkIndex: number;
  tokens: number;
  textExcerpt: string;
  /** Whether this chunk's vector is present in ChromaDB. */
  embedded: boolean;
  /** Dimensionality of the stored embedding, when present. */
  embeddingDim: number | null;
}

export interface DashboardMetrics {
  indexedDocuments: number;
  processingDocuments: number;
  /** Chunk rows in PostgreSQL — same source as DocumentFile.chunksCount. */
  vectorChunks: number;
  monthlyQueries: number;
  groundingRate: string;
  avgLatencyMs: number;
  precision: string;
  recall: string;
  f1Score: string;
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  userFeedbackCount?: number;
  positiveFeedbackCount?: number;
  negativeFeedbackCount?: number;
}

/** PostgreSQL ↔ ChromaDB reconciliation (GET /api/documents/index-health). */
export interface IndexHealth {
  chunkRows: number;
  vectors: number;
  inSync: boolean;
}

/** Editable metadata fields for a document (PATCH /api/documents/:id). */
export interface DocumentMetadataUpdate {
  name?: string;
  category?: string;
}
