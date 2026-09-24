import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatMessage, SourceChunk } from '../../models/chat.model';
import { RagService } from '../../core/services/rag.service';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  searchQuery = signal('');
  isLoading = signal(false);
  messages = signal<ChatMessage[]>([]);
  errorMessage = signal<string | null>(null);

  private ragService = inject(RagService);

  selectPrompt(promptText: string): void {
    this.searchQuery.set(promptText);
    this.onSearch();
  }

  onSearch(): void {
    const query = this.searchQuery().trim();
    if (!query || this.isLoading()) return;

    // Add user query to conversation
    this.messages.update((msgs) => [...msgs, { role: 'user', content: query }]);
    this.searchQuery.set('');
    this.isLoading.set(true);
    this.errorMessage.set(null);

    this.ragService.sendQuery(query).subscribe({
      next: (response) => {
        this.messages.update((msgs) => [
          ...msgs,
          {
            role: 'ai',
            content: response.content,
            sources: this.uniqueSources(response.sources),
            queryLogId: response.queryLogId,
            userFeedback: response.userFeedback ?? null,
          },
        ]);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error('RAG query failed:', err);
        this.errorMessage.set('Failed to get a response. Please try again.');
        this.isLoading.set(false);
      },
    });
  }

  submitFeedback(msg: ChatMessage, feedback: 'thumbs_up' | 'thumbs_down'): void {
    if (!msg.queryLogId || msg.userFeedback === feedback || msg.isFeedbackSubmitting) return;

    msg.isFeedbackSubmitting = true;
    this.messages.update((msgs) => [...msgs]);

    this.ragService.submitFeedback(msg.queryLogId, feedback).subscribe({
      next: () => {
        msg.userFeedback = feedback;
        msg.isFeedbackSubmitting = false;
        this.messages.update((msgs) => [...msgs]);
      },
      error: (err) => {
        console.error('Failed to submit feedback:', err);
        msg.isFeedbackSubmitting = false;
        this.messages.update((msgs) => [...msgs]);
      },
    });
  }

  /** One entry per source document — chunk positions and scores are not surfaced. */
  private uniqueSources(sources?: SourceChunk[]): SourceChunk[] {
    if (!sources) return [];

    const seen = new Set<string>();
    return sources.filter((s) => {
      if (seen.has(s.fileName)) return false;
      seen.add(s.fileName);
      return true;
    });
  }
}