export interface SourceChunk {
  id: string;
  fileName: string;
  chunkLocation: string;
  score: string;
}

/** A source the agentic pipeline cited and verified locally against retrieval. */
export interface AgentCitation {
  fileName: string;
  chunkLocation: string;
  similarity: number;
  bm25Score: number;
}

export type AgentMode = 'standard' | 'agentic';

export interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
  sources?: SourceChunk[];
  queryLogId?: string;
  userFeedback?: 'thumbs_up' | 'thumbs_down' | null;
  isFeedbackSubmitting?: boolean;

  /** Which pipeline produced this message. Absent on user messages. */
  agentMode?: AgentMode;

  // ── Agentic-only telemetry (populated from /api/agent/query) ──────────────
  citations?: AgentCitation[];
  /** 'greeting' answers bypass retrieval and generation entirely. */
  intent?: 'greeting' | 'retrieval';
  /** True only when a document-backed answer was actually produced. */
  answered?: boolean;
  /** Locally verified: citations resolve and the answer overlaps its context. */
  grounded?: boolean;
  confidenceScore?: number;
  /** Gemini generation calls spent. 0 for greetings and cache hits, else 1. */
  apiCallsUsed?: number;
  embeddingCallsUsed?: number;
  retryCount?: number;
  cached?: boolean;
  /** Ordered graph nodes visited, so the route taken is auditable in the UI. */
  graphPath?: string[];
  latencyMs?: number;
}
