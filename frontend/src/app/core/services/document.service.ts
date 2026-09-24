import { Injectable, inject, signal } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, timeout, catchError, throwError, tap, of } from 'rxjs';
import { environment } from '../../../environments/environment';
import {
  DocumentFile,
  VectorChunk,
  DashboardMetrics,
  DocumentMetadataUpdate,
  IndexHealth,
} from '../../models/document.model';

@Injectable({
  providedIn: 'root',
})
export class DocumentService {
  private http = inject(HttpClient);
  private apiUrl = environment.apiUrl;
  private requestTimeout = 60000;

  // ── SWR In-Memory Cache (Persists across route navigations) ──────────────────
  readonly cachedDocuments = signal<DocumentFile[]>([]);
  readonly cachedMetrics = signal<DashboardMetrics | null>(null);
  readonly cachedIndexHealth = signal<IndexHealth | null>(null);

  private lastDocsFetch = 0;
  private lastMetricsFetch = 0;
  private lastHealthFetch = 0;
  private readonly CACHE_TTL_MS = 30000; // 30 seconds

  invalidateCache(): void {
    this.lastDocsFetch = 0;
    this.lastMetricsFetch = 0;
    this.lastHealthFetch = 0;
  }

  getDocuments(force = false): Observable<DocumentFile[]> {
    const now = Date.now();
    if (!force && this.cachedDocuments().length > 0 && (now - this.lastDocsFetch < this.CACHE_TTL_MS)) {
      return of(this.cachedDocuments());
    }
    return this.http.get<DocumentFile[]>(`${this.apiUrl}/api/documents`).pipe(
      timeout(this.requestTimeout),
      tap((docs) => {
        this.cachedDocuments.set(docs);
        this.lastDocsFetch = Date.now();
      }),
      catchError(this.handleError)
    );
  }

  uploadDocument(file: File, category: string): Observable<DocumentFile> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('category', category);
    return this.http.post<DocumentFile>(`${this.apiUrl}/api/documents/upload`, formData).pipe(
      timeout(300000),
      tap((newDoc) => {
        this.cachedDocuments.update((docs) => [newDoc, ...docs.filter((d) => d.id !== newDoc.id)]);
        this.invalidateCache();
      }),
      catchError(this.handleError)
    );
  }

  updateDocument(id: string, changes: DocumentMetadataUpdate): Observable<DocumentFile> {
    return this.http.patch<DocumentFile>(`${this.apiUrl}/api/documents/${id}`, changes).pipe(
      timeout(this.requestTimeout),
      tap((updated) => {
        this.cachedDocuments.update((docs) => docs.map((d) => (d.id === updated.id ? updated : d)));
        this.invalidateCache();
      }),
      catchError(this.handleError)
    );
  }

  deleteDocument(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/api/documents/${id}`).pipe(
      timeout(this.requestTimeout),
      tap(() => {
        this.cachedDocuments.update((docs) => docs.filter((d) => d.id !== id));
        this.invalidateCache();
      }),
      catchError(this.handleError)
    );
  }

  /** Re-chunks, re-embeds and replaces the document's vectors. */
  reindexDocument(id: string): Observable<DocumentFile> {
    return this.http.post<DocumentFile>(`${this.apiUrl}/api/documents/${id}/reindex`, {}).pipe(
      timeout(300000),
      tap((updated) => {
        this.cachedDocuments.update((docs) => docs.map((d) => (d.id === updated.id ? updated : d)));
        this.invalidateCache();
      }),
      catchError(this.handleError)
    );
  }

  getChunks(docId: string): Observable<VectorChunk[]> {
    return this.http.get<VectorChunk[]>(`${this.apiUrl}/api/documents/${docId}/chunks`).pipe(
      timeout(this.requestTimeout),
      catchError(this.handleError)
    );
  }

  /** Chunk rows in PostgreSQL vs vectors in ChromaDB. */
  getIndexHealth(force = false): Observable<IndexHealth> {
    const now = Date.now();
    if (!force && this.cachedIndexHealth() && (now - this.lastHealthFetch < this.CACHE_TTL_MS)) {
      return of(this.cachedIndexHealth()!);
    }
    return this.http.get<IndexHealth>(`${this.apiUrl}/api/documents/index-health`).pipe(
      timeout(this.requestTimeout),
      tap((h) => {
        this.cachedIndexHealth.set(h);
        this.lastHealthFetch = Date.now();
      }),
      catchError(this.handleError)
    );
  }

  getMetrics(force = false): Observable<DashboardMetrics> {
    const now = Date.now();
    if (!force && this.cachedMetrics() && (now - this.lastMetricsFetch < this.CACHE_TTL_MS)) {
      return of(this.cachedMetrics()!);
    }
    const url = force ? `${this.apiUrl}/api/metrics?refresh=true` : `${this.apiUrl}/api/metrics`;
    return this.http.get<DashboardMetrics>(url).pipe(
      timeout(this.requestTimeout),
      tap((m) => {
        this.cachedMetrics.set(m);
        this.lastMetricsFetch = Date.now();
      }),
      catchError(this.handleError)
    );
  }

  private handleError(error: any) {
    console.error('API Error:', error);
    if (error.name === 'TimeoutError') {
      return throwError(() => new Error('Request timed out. Please try again.'));
    }
    if (error instanceof HttpErrorResponse) {
      return throwError(() => new Error(error.error?.detail || error.message || 'An error occurred'));
    }
    return throwError(() => error);
  }
}
