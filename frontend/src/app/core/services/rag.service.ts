import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, timeout, catchError, throwError } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AgentMode, ChatMessage } from '../../models/chat.model';

export interface QueryRequest {
  query: string;
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
  private apiUrl = environment.apiUrl;
  private requestTimeout = 120000;

  /** Baseline RAG pipeline — rag_service on port 8002. Unchanged. */
  sendQuery(query: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.apiUrl}/api/chat/query`, {
      query,
    } satisfies QueryRequest).pipe(
      timeout(this.requestTimeout),
      catchError(this.handleError)
    );
  }

  /**
   * Agentic RAG pipeline — LangGraph agentic_rag_service on port 8005.
   * Returns the same ChatMessage shape plus agentic telemetry fields.
   */
  queryAgent(query: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.apiUrl}/api/agent/query`, {
      query,
    } satisfies QueryRequest).pipe(
      timeout(this.requestTimeout),
      catchError(this.handleError)
    );
  }

  /** Dispatch to whichever pipeline the chat header toggle has selected. */
  sendQueryForMode(query: string, mode: AgentMode): Observable<ChatMessage> {
    return mode === 'agentic' ? this.queryAgent(query) : this.sendQuery(query);
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
    console.error('RAG API Error:', error);
    if (error.name === 'TimeoutError') {
      return throwError(() => new Error('Query processing timed out. Please try again.'));
    }
    if (error instanceof HttpErrorResponse) {
      return throwError(() => new Error(error.error?.detail || error.message || 'Failed to process query'));
    }
    return throwError(() => error);
  }
}
