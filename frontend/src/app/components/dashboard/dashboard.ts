import { Component, computed, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  CategoryItem,
  DashboardMetrics,
  DocumentFile,
  IndexHealth,
  VectorChunk,
} from '../../models/document.model';
import { DocumentService } from '../../core/services/document.service';
import { ToastService } from '../../core/services/toast.service';
import { LoggerService } from '../../core/services/logger.service';
import { formatUserError } from '../../core/utils/error-formatter';

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
  // Document Filtering Signals
  nameFilter = signal('');
  categoryFilter = signal('all');
  startDateFilter = signal('');
  endDateFilter = signal('');
  filterQuery = this.nameFilter; // backward-compat alias

  selectedDocForDrawer = signal<DocumentFile | null>(null);
  drawerChunks = signal<VectorChunk[]>([]);
  isDrawerLoading = signal(false);

  // Dynamic Knowledge Categories
  categories = signal<CategoryItem[]>([]);
  isLoadingCategories = signal(false);
  isCategoryModalOpen = signal(false);
  newCategoryName = signal('');
  newCategoryDescription = signal('');
  isCreatingCategory = signal(false);
  categoryActionError = signal<string | null>(null);
  deletingCategoryId = signal<string | null>(null);

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
  private toastService = inject(ToastService);
  private logger = inject(LoggerService);

  ngOnInit(): void {
    // 1. Instant hydration from persistent SWR cache (0ms delay, zero spinners)
    const cachedDocs = this.documentService.cachedDocuments();
    if (cachedDocs.length > 0) {
      this.documents.set(cachedDocs);
      this.isLoadingDocs.set(false);
    }
    const cachedCats = this.documentService.cachedCategories();
    if (cachedCats.length > 0) {
      this.categories.set(cachedCats);
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
    this.loadCategories(false);
    this.loadMetrics(false);
    this.loadIndexHealth(false);
  }

  readonly hasActiveFilters = computed(() =>
    Boolean(
      this.nameFilter().trim() ||
      this.categoryFilter() !== 'all' ||
      this.startDateFilter() ||
      this.endDateFilter()
    )
  );

  readonly filteredDocuments = computed(() => {
    const nameQ = this.nameFilter().trim().toLowerCase();
    const cat = this.categoryFilter();
    const start = this.startDateFilter();
    const end = this.endDateFilter();
    const docs = this.documents();

    return docs.filter((d) => {
      // 1. Filter by file name (case-insensitive substring)
      if (nameQ && !d.name.toLowerCase().includes(nameQ)) {
        return false;
      }

      // 2. Filter by category
      if (cat !== 'all' && d.category !== cat) {
        return false;
      }

      // 3. Filter by date range (inclusive)
      if (start || end) {
        const rawDate = d.rawDate || d.lastUpdated;
        const time = new Date(rawDate).getTime();
        if (!isNaN(time)) {
          if (start) {
            const startMs = new Date(`${start}T00:00:00`).getTime();
            if (time < startMs) return false;
          }
          if (end) {
            const endMs = new Date(`${end}T23:59:59.999`).getTime();
            if (time > endMs) return false;
          }
        }
      }

      return true;
    });
  });

  resetFilters(): void {
    this.nameFilter.set('');
    this.categoryFilter.set('all');
    this.startDateFilter.set('');
    this.endDateFilter.set('');
  }

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
    this.avgLatencyMs.set(m.avgLatencyMs);
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
    this.logger.debug('Loading documents', { force });
    this.documentService.getDocuments(force).subscribe({
      next: (docs) => {
        this.logger.debug(`Loaded ${docs.length} documents`);
        this.documents.set(docs);
        this.isLoadingDocs.set(false);
      },
      error: (err) => {
        this.logger.error('Failed to load documents:', err);
        this.isLoadingDocs.set(false);
      },
    });
  }

  private loadMetrics(force = false): void {
    this.documentService.getMetrics(force).subscribe({
      next: (m) => this.applyMetrics(m),
      error: (err) => this.logger.error('Failed to load metrics:', err),
    });
  }

  private loadIndexHealth(force = false): void {
    this.documentService.getIndexHealth(force).subscribe({
      next: (h) => this.applyIndexHealth(h),
      error: (err) => {
        this.logger.error('Failed to load index health:', err);
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
    this.logger.info('Refreshing dashboard data', { force });
    this.isRefreshing.set(true);
    this.documentService.getDocuments(force).subscribe({
      next: (docs) => {
        this.documents.set(docs);
        this.isRefreshing.set(false);
        this.syncOpenDrawer(docs);
      },
      error: (err) => {
        this.logger.error('Failed to refresh documents:', err);
        this.isRefreshing.set(false);
      },
    });
    this.loadCategories(force);
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

  // Toast Notification Helper (delegates to the unified global ToastService)
  showNotification(msg: string): void {
    const lower = msg.toLowerCase();
    const isError =
      lower.startsWith('error') ||
      lower.includes('failed') ||
      lower.includes('could not') ||
      lower.includes('resource_exhausted');

    if (isError) {
      const formatted = formatUserError(msg);
      this.toastService.error(formatted.message, formatted.title);
    } else {
      this.toastService.success(msg);
    }
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
      this.toastService.warning(
        'Please select a supported file format: PDF, Word (DOCX), Text, or Excel (XLSX).',
        'Unsupported File'
      );
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
    this.logger.info(`Starting upload of "${file.name}"`, {
      size: file.size,
      category: this.selectedCategory,
    });

    this.documentService.uploadDocument(file, this.selectedCategory).subscribe({
      next: (newDoc) => {
        this.logger.info(`Uploaded document successfully`, {
          id: newDoc.id,
          name: newDoc.name,
          chunks: newDoc.chunksCount,
        });
        this.isUploadModalOpen.set(false);
        this.isUploading.set(false);
        this.showNotification(`Document "${newDoc.name}" uploaded successfully!`);
        // Re-read list + metrics rather than patching the list locally, so the
        // chunk counts on the cards and in the table come from one snapshot.
        this.refreshDashboard();
      },
      error: (err) => {
        this.logger.error('Upload failed:', err);
        this.isUploading.set(false);
      },
    });
  }

  // Chunk Drawer Handlers
  openChunkDrawer(doc: DocumentFile): void {
    this.logger.info(`Opening chunk drawer for document ${doc.id} ("${doc.name}")`);
    this.selectedDocForDrawer.set(doc);
    this.drawerChunks.set([]);
    this.isDrawerLoading.set(true);

    this.documentService.getChunks(doc.id).subscribe({
      next: (chunks) => {
        this.logger.debug(`Loaded ${chunks.length} chunks for document ${doc.id}`);
        this.drawerChunks.set(chunks);
        this.isDrawerLoading.set(false);
      },
      error: (err) => {
        this.logger.error('Failed to load chunks:', err);
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

  /** Category list for the edit modal, populated from dynamic categories. */
  readonly editCategoryOptions = computed<CategoryOption[]>(() => {
    const cats = this.categories().map((c) => ({ value: c.name, label: c.name }));
    const current = this.docPendingEdit()?.category;
    if (!current || cats.some((c) => c.value === current)) {
      return cats.length > 0 ? cats : [{ value: 'Security & Policy', label: 'Security & Policy' }];
    }
    return [...cats, { value: current, label: current }];
  });

  // ── Category Management Handlers ──────────────────────────────────────────

  loadCategories(force = false): void {
    this.isLoadingCategories.set(true);
    this.documentService.getCategories(force).subscribe({
      next: (cats) => {
        this.categories.set(cats);
        this.isLoadingCategories.set(false);
        if (cats.length > 0 && !cats.some((c) => c.name === this.selectedCategory)) {
          this.selectedCategory = cats[0].name;
        }
      },
      error: (err) => {
        this.logger.error('Failed to load categories:', err);
        this.isLoadingCategories.set(false);
      },
    });
  }

  openCategoryModal(): void {
    this.newCategoryName.set('');
    this.newCategoryDescription.set('');
    this.categoryActionError.set(null);
    this.isCategoryModalOpen.set(true);
    this.loadCategories(true);
  }

  closeCategoryModal(): void {
    if (this.isCreatingCategory()) return;
    this.isCategoryModalOpen.set(false);
    this.categoryActionError.set(null);
  }

  createCategory(): void {
    const name = this.newCategoryName().trim();
    const desc = this.newCategoryDescription().trim();
    if (!name || this.isCreatingCategory()) return;

    this.isCreatingCategory.set(true);
    this.categoryActionError.set(null);

    this.documentService.createCategory(name, desc).subscribe({
      next: (created) => {
        this.toastService.success(`Category "${created.name}" created successfully.`);
        this.newCategoryName.set('');
        this.newCategoryDescription.set('');
        this.isCreatingCategory.set(false);
        this.loadCategories(true);
      },
      error: (err) => {
        const errorMsg = err?.message || 'Failed to create category.';
        this.categoryActionError.set(errorMsg);
        this.toastService.error(errorMsg, 'Category Creation Failed');
        this.isCreatingCategory.set(false);
      },
    });
  }

  deleteCategory(cat: CategoryItem): void {
    if (cat.documentCount > 0) {
      this.toastService.warning(
        `Cannot delete "${cat.name}" because ${cat.documentCount} document(s) are assigned to it.`,
        'Category In Use'
      );
      return;
    }

    if (this.deletingCategoryId()) return;
    this.deletingCategoryId.set(cat.id);
    this.categoryActionError.set(null);

    this.documentService.deleteCategory(cat.id).subscribe({
      next: () => {
        this.toastService.success(`Category "${cat.name}" deleted successfully.`);
        this.deletingCategoryId.set(null);
        this.loadCategories(true);
      },
      error: (err) => {
        const errorMsg = err?.message || `Failed to delete category "${cat.name}".`;
        this.categoryActionError.set(errorMsg);
        this.toastService.error(errorMsg, 'Category Deletion Failed');
        this.deletingCategoryId.set(null);
      },
    });
  }

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
    const newName = this.editName().trim();
    const newCategory = this.editCategory();
    this.logger.info(`Saving metadata update for doc ${doc.id}`, { newName, newCategory });

    this.documentService
      .updateDocument(doc.id, { name: newName, category: newCategory })
      .subscribe({
        next: (updated) => {
          this.logger.info(`Metadata updated successfully for doc ${updated.id}`);
          this.isSavingEdit.set(false);
          this.docPendingEdit.set(null);
          this.showNotification(`Metadata updated for "${updated.name}".`);
          this.refreshDashboard();
        },
        error: (err) => {
          this.logger.error('Metadata update failed:', err);
          this.isSavingEdit.set(false);
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
    this.logger.info(`Initiating re-index for doc ${doc.id} ("${doc.name}")`);
    this.showNotification(`Re-indexing "${doc.name}"…`);

    this.documentService.reindexDocument(doc.id).subscribe({
      next: (updated) => {
        this.logger.info(`Re-indexed doc ${updated.id} successfully`, {
          chunksCount: updated.chunksCount,
        });
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
        this.logger.error('Reindex failed:', err);
        this.setReindexing(doc.id, false);
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
    this.logger.info(`Deleting document ${doc.id} ("${doc.name}")`);

    this.documentService.deleteDocument(doc.id).subscribe({
      next: () => {
        this.logger.info(`Document ${doc.id} successfully deleted`);
        this.showNotification(`Removed "${doc.name}" from knowledge base.`);
        this.refreshDashboard();
      },
      error: (err) => {
        this.logger.error('Delete failed:', err);
      },
    });
  }
}
