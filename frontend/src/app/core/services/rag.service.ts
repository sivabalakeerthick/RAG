import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, timeout, catchError, throwError } from 'rxjs';
import { environment } from '../../../environments/environment';
import { ChatMessage } from '../../models/chat.model';

export interface QueryRequest {
  query: string;
}

@Injectable({
  providedIn: 'root',
})
export class RagService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;
  private requestTimeout = 120000;

  sendQuery(query: string): Observable<ChatMessage> {
    return this.http.post<ChatMessage>(`${this.apiUrl}/api/chat/query`, {
      query,
    } satisfies QueryRequest).pipe(
      timeout(this.requestTimeout),
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
