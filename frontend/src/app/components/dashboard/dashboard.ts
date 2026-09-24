import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  DashboardMetrics,
  DocumentFile,
  IndexHealth,
  VectorChunk,
} from '../../models/document.model';
import { DocumentService } from '../../core/services/document.service';

interface CategoryOption {
  value: string;
  label: string;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class Dashboard implements OnInit {
  filterQuery = signal('');
  selectedDocForDrawer = signal<DocumentFile | null>(null);
  drawerChunks = signal<VectorChunk[]>([]);
  isDrawerLoading = signal(false);

  // Notification Toast State
  notificationMessage = signal<string | null>(null);

  // Knowledge categories — shared by the upload and edit-metadata modals
  readonly categories: CategoryOption[] = [
    { value: 'Security & Policy', label: 'Security & Compliance' },
    { value: 'IT Support', label: 'IT & Infrastructure' },
    { value: 'Engineering', label: 'Engineering & API Specs' },
    { value: 'HR & Operations', label: 'HR & Legal Policies' },
  ];

  // Inline Upload Modal State
  isUploadModalOpen = signal(false);
  isUploading = signal(false);
  selectedFile = signal<File | null>(null);
  selectedCategory = 'Security & Policy';
  isDragging = signal(false);

  // Edit Metadata Modal State
  docPendingEdit = signal<DocumentFile | null>(null);
  editName = signal('');
  editCategory = signal('');
  isSavingEdit = signal(false);

  // Delete Confirm Modal State
  isDeleteConfirmOpen = signal(false);
  docPendingDelete = signal<DocumentFile | null>(null);

  // Per-row re-index state — keyed by document id so a row's button can be
  // disabled while its request is in flight (a second click used to fire a
  // second concurrent re-index and duplicate the chunks).
  reindexingIds = signal<Set<string>>(new Set());

  // Dashboard metrics (loaded from API)
  monthlyQueries = signal(0);
  groundingRate = signal('—');
  avgLatencyMs = signal(0);
  precision = signal('—');
  recall = signal('—');
  f1Score = signal('—');
  tp = signal(0);
  tn = signal(0);
  fp = signal(0);
  fn = signal(0);
  userFeedbackCount = signal(0);
  positiveFeedbackCount = signal(0);
  negativeFeedbackCount = signal(0);
  // PostgreSQL ↔ ChromaDB reconciliation
  storeVectors = signal(0);
  storeChunkRows = signal(0);
  isStoreInSync = signal(true);
  hasIndexHealth = signal(false);

  documents = signal<DocumentFile[]>([]);
  isLoadingDocs = signal(true);
  isRefreshing = signal(false);

  private documentService = inject(DocumentService);

  ngOnInit(): void {
    // 1. Instant hydration from persistent SWR cache (0ms delay, zero spinners)
    const cachedDocs = this.documentService.cachedDocuments();
    if (cachedDocs.length > 0) {
      this.documents.set(cachedDocs);
      this.isLoadingDocs.set(false);
    }
    const cachedMetrics = this.documentService.cachedMetrics();
    if (cachedMetrics) {
      this.applyMetrics(cachedMetrics);
    }
    const cachedHealth = this.documentService.cachedIndexHealth();
    if (cachedHealth) {
      this.applyIndexHealth(cachedHealth);
    }

    // 2. Silent background revalidation (or first-time fetch)
    this.loadDocuments(false);
    this.loadMetrics(false);
    this.loadIndexHealth(false);
  }

  readonly filteredDocuments = computed(() => {
    const q = this.filterQuery().trim().toLowerCase();
    const docs = this.documents();
    if (!q) return docs;
    return docs.filter(
      (d) =>
        d.name.toLowerCase().includes(q) ||
        d.category.toLowerCase().includes(q) ||
        d.status.toLowerCase().includes(q)
    );
  });

  // ── Derived metrics ────────────────────────────────────────────────────────
  // The chunk total is summed from the same document list the table renders,
  // so the "Vector Chunks" card can never disagree with the per-row counts.

  readonly totalChunks = computed(() =>
    this.documents().reduce((sum, d) => sum + d.chunksCount, 0)
  );

  readonly indexedCount = computed(
    () => this.documents().filter((d) => d.status === 'Indexed').length
  );

  readonly processingCount = computed(
    () => this.documents().filter((d) => d.status === 'Processing').length
  );

  /**
   * True when the three chunk counts disagree: the total shown in the table,
   * all chunk rows in PostgreSQL (would differ if rows were orphaned), and the
   * vectors in ChromaDB.
   */
  readonly isVectorStoreOutOfSync = computed(
    () =>
      this.hasIndexHealth() &&
      (!this.isStoreInSync() || this.storeChunkRows() !== this.totalChunks())
  );

  // ── Loading / refreshing ───────────────────────────────────────────────────

  private applyMetrics(m: DashboardMetrics): void {
    this.monthlyQueries.set(m.monthlyQueries);
    this.groundingRate.set(m.groundingRate);
    this.avgLatencyMs.set(m.avgLatencyMs / 10);
    this.precision.set(m.precision || '—');
    this.recall.set(m.recall || '—');
    this.f1Score.set(m.f1Score || '—');
    this.tp.set(m.tp || 0);
    this.tn.set(m.tn || 0);
    this.fp.set(m.fp || 0);
    this.fn.set(m.fn || 0);
    this.userFeedbackCount.set(m.userFeedbackCount || 0);
    this.positiveFeedbackCount.set(m.positiveFeedbackCount || 0);
    this.negativeFeedbackCount.set(m.negativeFeedbackCount || 0);
  }

  private applyIndexHealth(h: IndexHealth): void {
    this.storeChunkRows.set(h.chunkRows);
    this.storeVectors.set(h.vectors);
    this.isStoreInSync.set(h.inSync);
    this.hasIndexHealth.set(true);
  }

  private loadDocuments(force = false): void {
    if (this.documents().length === 0) {
      this.isLoadingDocs.set(true);
    }
    this.documentService.getDocuments(force).subscribe({
      next: (docs) => {
        this.documents.set(docs);
        this.isLoadingDocs.set(false);
      },
      error: (err) => {
        console.error('Failed to load documents:', err);
        if (this.documents().length === 0) {
          this.showNotification(`Could not load documents: ${err.message ?? 'unknown error'}`);
        }
        this.isLoadingDocs.set(false);
      },
    });
  }

  private loadMetrics(force = false): void {
    this.documentService.getMetrics(force).subscribe({
      next: (m) => this.applyMetrics(m),
      error: (err) => console.error('Failed to load metrics:', err),
    });
  }

  private loadIndexHealth(force = false): void {
    this.documentService.getIndexHealth(force).subscribe({
      next: (h) => this.applyIndexHealth(h),
      error: (err) => {
        console.error('Failed to load index health:', err);
        this.hasIndexHealth.set(false);
      },
    });
  }

  /**
   * Re-reads the document list, the metrics and the index health together.
   * Called after every mutation (upload / edit / re-index / delete) so the stat
   * cards, the table and the vector store never drift apart.
   */
  refreshDashboard(force = true): void {
    this.isRefreshing.set(true);
    this.documentService.getDocuments(force).subscribe({
      next: (docs) => {
        this.documents.set(docs);
        this.isRefreshing.set(false);
        this.syncOpenDrawer(docs);
      },
      error: (err) => {
        console.error('Failed to refresh documents:', err);
        this.isRefreshing.set(false);
      },
    });
    this.loadMetrics(force);
    this.loadIndexHealth(force);
  }

  /** Keeps the open chunk drawer pointed at the refreshed document record. */
  private syncOpenDrawer(docs: DocumentFile[]): void {
    const open = this.selectedDocForDrawer();
    if (!open) return;
    const fresh = docs.find((d) => d.id === open.id);
    if (fresh) {
      this.selectedDocForDrawer.set(fresh);
    } else {
      this.closeChunkDrawer();
    }
  }

  // Toast Notification Helper (auto-dismiss after 2 seconds)
  showNotification(msg: string): void {
    this.notificationMessage.set(msg);
    setTimeout(() => {
      this.notificationMessage.set(null);
    }, 2000);
  }

  // Upload Modal Handlers
  openUploadModal(): void {
    this.selectedFile.set(null);
    this.isUploadModalOpen.set(true);
  }

  closeUploadModal(): void {
    if (this.isUploading()) return;
    this.isUploadModalOpen.set(false);
  }

  // Drag & Drop Handlers
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragging.set(false);
    if (event.dataTransfer?.files.length) {
      this.handleFile(event.dataTransfer.files[0]);
    }
  }

  onFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement;
    if (target.files?.length) {
      this.handleFile(target.files[0]);
    }
  }

  private handleFile(file: File): void {
    const validTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
      'application/vnd.ms-excel',                                           // .xls
    ];
    // Some browsers report application/octet-stream for xlsx — allow by extension too
    const ext = file.name.split('.').pop()?.toLowerCase();
    const validExts = ['pdf', 'docx', 'doc', 'txt', 'xlsx', 'xls'];
    if (!validTypes.includes(file.type) && !validExts.includes(ext ?? '')) {
      this.showNotification('Error: Please upload a PDF, DOCX, TXT, XLSX, or XLS file.');
      return;
    }
    this.selectedFile.set(file);
  }

  removeFile(event: Event): void {
    event.stopPropagation();
    this.selectedFile.set(null);
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /** Returns the Font Awesome icon class based on the file extension */
  getFileIcon(fileName: string): string {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':  return 'fa-solid fa-file-pdf';
      case 'docx':
      case 'doc':  return 'fa-solid fa-file-word';
      case 'txt':  return 'fa-solid fa-file-lines';
      case 'xlsx':
      case 'xls':  return 'fa-solid fa-file-excel';
      default:     return 'fa-solid fa-file';
    }
  }

  /** Returns the icon colour class based on file extension */
  getFileIconColor(fileName: string): string {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':  return 'pdf-icon';
      case 'docx':
      case 'doc':  return 'word-icon';
      case 'txt':  return 'txt-icon';
      case 'xlsx':
      case 'xls':  return 'excel-icon';
      default:     return 'default-icon';
    }
  }

  startUpload(): void {
    const file = this.selectedFile();
    if (!file || this.isUploading()) return;
    this.isUploading.set(true);

    this.documentService.uploadDocument(file, this.selectedCategory).subscribe({
      next: (newDoc) => {
        this.isUploadModalOpen.set(false);
        this.isUploading.set(false);
        this.showNotification(`Document "${newDoc.name}" uploaded successfully!`);
        // Re-read list + metrics rather than patching the list locally, so the
        // chunk counts on the cards and in the table come from one snapshot.
        this.refreshDashboard();
      },
      error: (err) => {
        console.error('Upload failed:', err);
        this.isUploading.set(false);
        this.showNotification(`Upload failed: ${err.message ?? 'unknown error'}`);
      },
    });
  }

  // Chunk Drawer Handlers
  openChunkDrawer(doc: DocumentFile): void {
    this.selectedDocForDrawer.set(doc);
    this.drawerChunks.set([]);
    this.isDrawerLoading.set(true);

    this.documentService.getChunks(doc.id).subscribe({
      next: (chunks) => {
        this.drawerChunks.set(chunks);
        this.isDrawerLoading.set(false);
      },
      error: (err) => {
        console.error('Failed to load chunks:', err);
        this.isDrawerLoading.set(false);
        this.showNotification('Error: Could not load vector chunks.');
      },
    });
  }

  closeChunkDrawer(): void {
    this.selectedDocForDrawer.set(null);
    this.drawerChunks.set([]);
  }

  copyChunk(text: string): void {
    navigator.clipboard.writeText(text);
    this.showNotification('Chunk excerpt copied to clipboard!');
  }

  /** Chunks in the drawer that are missing their vector in ChromaDB. */
  readonly drawerMissingVectors = computed(
    () => this.drawerChunks().filter((c) => !c.embedded).length
  );

  // ── Edit Metadata ─────────────────────────────────────────────────────────

  editDoc(doc: DocumentFile): void {
    this.docPendingEdit.set(doc);
    this.editName.set(doc.name);
    this.editCategory.set(doc.category);
  }

  closeEditModal(): void {
    if (this.isSavingEdit()) return;
    this.docPendingEdit.set(null);
  }

  /** Category list for the edit modal, including any value not in the presets. */
  readonly editCategoryOptions = computed<CategoryOption[]>(() => {
    const current = this.docPendingEdit()?.category;
    if (!current || this.categories.some((c) => c.value === current)) {
      return this.categories;
    }
    return [...this.categories, { value: current, label: current }];
  });

  readonly isEditDirty = computed(() => {
    const doc = this.docPendingEdit();
    if (!doc) return false;
    const name = this.editName().trim();
    return name.length > 0 && (name !== doc.name || this.editCategory() !== doc.category);
  });

  saveEdit(): void {
    const doc = this.docPendingEdit();
    if (!doc || this.isSavingEdit() || !this.isEditDirty()) return;

    this.isSavingEdit.set(true);
    this.documentService
      .updateDocument(doc.id, { name: this.editName().trim(), category: this.editCategory() })
      .subscribe({
        next: (updated) => {
          this.isSavingEdit.set(false);
          this.docPendingEdit.set(null);
          this.showNotification(`Metadata updated for "${updated.name}".`);
          this.refreshDashboard();
        },
        error: (err) => {
          console.error('Metadata update failed:', err);
          this.isSavingEdit.set(false);
          this.showNotification(`Error: ${err.message ?? 'could not update metadata'}`);
        },
      });
  }

  // ── Re-index ──────────────────────────────────────────────────────────────

  isReindexing(docId: string): boolean {
    return this.reindexingIds().has(docId);
  }

  private setReindexing(docId: string, active: boolean): void {
    this.reindexingIds.update((ids) => {
      const next = new Set(ids);
      if (active) {
        next.add(docId);
      } else {
        next.delete(docId);
      }
      return next;
    });
  }

  reindexDoc(doc: DocumentFile): void {
    if (this.isReindexing(doc.id)) return;
    this.setReindexing(doc.id, true);
    this.showNotification(`Re-indexing "${doc.name}"…`);

    this.documentService.reindexDocument(doc.id).subscribe({
      next: (updated) => {
        this.setReindexing(doc.id, false);
        this.showNotification(
          `Re-indexed "${updated.name}" — ${updated.chunksCount} chunks.`
        );
        this.refreshDashboard();
        if (this.selectedDocForDrawer()?.id === doc.id) {
          this.openChunkDrawer(updated);
        }
      },
      error: (err) => {
        console.error('Reindex failed:', err);
        this.setReindexing(doc.id, false);
        this.showNotification(`Error: ${err.message ?? `failed to re-index ${doc.name}`}`);
        this.refreshDashboard();
      },
    });
  }

  // Delete flow using custom modal (replaces window.confirm)
  requestDelete(doc: DocumentFile): void {
    this.docPendingDelete.set(doc);
    this.isDeleteConfirmOpen.set(true);
  }

  cancelDelete(): void {
    this.isDeleteConfirmOpen.set(false);
    this.docPendingDelete.set(null);
  }

  confirmDelete(): void {
    const doc = this.docPendingDelete();
    if (!doc) return;

    this.isDeleteConfirmOpen.set(false);
    this.docPendingDelete.set(null);

    this.documentService.deleteDocument(doc.id).subscribe({
      next: () => {
        this.showNotification(`Removed "${doc.name}" from knowledge base.`);
        this.refreshDashboard();
      },
      error: (err) => {
        console.error('Delete failed:', err);
        this.showNotification(`Error: Failed to delete "${doc.name}".`);
        this.refreshDashboard();
      },
    });
  }
}
