import { Injectable, inject, signal } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, timeout, catchError, throwError } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AgentMode, ChatMessage } from '../../models/chat.model';
import { formatUserError } from '../utils/error-formatter';
import { LoggerService } from './logger.service';

export interface QueryRequest {
  query: string;
  category?: string;
  document_id?: string;
  session_id?: string;
}

/** Response shape of GET /api/agent/health. */
export interface AgentHealth {
  status: string;
  service: string;
  port: number;
  model: string;
  maxLlmCallsPerQuery: number;
  maxRetries: number;
  retrieveTopK: number;
  rateLimiter: { used: number; limit: number; remaining: number; windowSeconds: number };
  cache: { entries: number; hits: number; misses: number; ttlSeconds: number };
}

@Injectable({
  providedIn: 'root',
})
export class RagService {
  private http = inject(HttpClient);
  private logger = inject(LoggerService);
  private apiUrl = environment.apiUrl;
  private requestTimeout = 120000;

  private readonly MESSAGES_KEY = 'cognidoc_chat_messages';
  private readonly SESSION_KEY = 'cognidoc_active_session';
  private readonly MODE_KEY = 'cognidoc_active_mode';

  // ── Singleton conversation state (survives route navigation across components) ──
  messages = signal<ChatMessage[]>(this.loadStoredMessages());
  sessionId = signal<string>(this.loadStoredSessionId());
  activeMode = signal<AgentMode>(this.loadStoredMode());

  setMode(mode: AgentMode): void {
    this.activeMode.set(mode);
    try {
      sessionStorage.setItem(this.MODE_KEY, mode);
    } catch {}
  }

  addMessage(msg: ChatMessage): void {
    this.messages.update((msgs) => {
      const updated = [...msgs, msg];
      try {
        sessionStorage.setItem(this.MESSAGES_KEY, JSON.stringify(updated));
      } catch {}
      return updated;
    });
  }

  clearChat(): void {
    this.messages.set([]);
    const newId = crypto.randomUUID();
    this.sessionId.set(newId);
    try {
      sessionStorage.removeItem(this.MESSAGES_KEY);
      sessionStorage.setItem(this.SESSION_KEY, newId);
    } catch {}
    this.logger.info('Conversation history cleared and new session started');
  }

  updateMessageFeedback(queryLogId: string, feedback: 'thumbs_up' | 'thumbs_down'): void {
    this.messages.update((msgs) => {
      const updated = msgs.map((m) =>
        m.queryLogId === queryLogId ? { ...m, userFeedback: feedback, isFeedbackSubmitting: false } : m
      );
      try {
        sessionStorage.setItem(this.MESSAGES_KEY, JSON.stringify(updated));
      } catch {}
      return updated;
    });
  }

  private loadStoredMessages(): ChatMessage[] {
    try {
      const saved = sessionStorage.getItem(this.MESSAGES_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  }

  private loadStoredSessionId(): string {
    try {
      const saved = sessionStorage.getItem(this.SESSION_KEY);
      if (saved) return saved;
      const newId = crypto.randomUUID();
      sessionStorage.setItem(this.SESSION_KEY, newId);
      return newId;
    } catch {
      return crypto.randomUUID();
    }
  }

  private loadStoredMode(): AgentMode {
    try {
      const saved = sessionStorage.getItem(this.MODE_KEY) as AgentMode;
      return (saved === 'standard' || saved === 'agentic') ? saved : 'standard';
    } catch {
      return 'standard';
    }
  }

  /** Baseline RAG pipeline — rag_service on port 8002. */
  sendQuery(query: string, category?: string, documentId?: string, sessionId?: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.apiUrl}/api/chat/query`, {
      query,
      category,
      document_id: documentId,
      session_id: sessionId,
    } satisfies QueryRequest).pipe(
      timeout(this.requestTimeout),
      catchError(this.handleError)
    );
  }

  /**
   * Agentic RAG pipeline — LangGraph agentic_rag_service on port 8005.
   * Returns the same ChatMessage shape plus agentic telemetry fields.
   */
  queryAgent(query: string, category?: string, documentId?: string, sessionId?: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.apiUrl}/api/agent/query`, {
      query,
      category,
      document_id: documentId,
      session_id: sessionId,
    } satisfies QueryRequest).pipe(
      timeout(this.requestTimeout),
      catchError(this.handleError)
    );
  }

  /** Dispatch to whichever pipeline the chat header toggle has selected. */
  sendQueryForMode(query: string, mode: AgentMode, sessionId?: string, category?: string, documentId?: string): Observable<ChatMessage> {
    return mode === 'agentic' ? this.queryAgent(query, category, documentId, sessionId) : this.sendQuery(query, category, documentId, sessionId);
  }

  /** Live budget + cache telemetry from the agentic service. */
  getAgentHealth(): Observable<AgentHealth> {
    return this.http.get<AgentHealth>(`${this.apiUrl}/api/agent/health`).pipe(
      timeout(15000),
      catchError(this.handleError)
    );
  }

  submitFeedback(queryLogId: string, feedback: 'thumbs_up' | 'thumbs_down'): Observable<any> {
    return this.http.post(`${this.apiUrl}/api/chat/feedback`, {
      query_log_id: queryLogId,
      feedback,
    }).pipe(
      timeout(15000),
      catchError(this.handleError)
    );
  }

  private handleError(error: any) {
    this.logger.error('RAG', 'Query processing error:', error);
    if (error.name === 'TimeoutError') {
      return throwError(() => new Error('Query processing timed out. Please try again.'));
    }
    if (error instanceof HttpErrorResponse) {
      const formatted = formatUserError(error, 'Failed to process query. Please try again.');
      return throwError(() => new Error(formatted.message));
    }
    return throwError(() => error);
  }
}
