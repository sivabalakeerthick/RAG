import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AgentMode, ChatMessage, SourceChunk } from '../../models/chat.model';
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

  /**
   * Which pipeline the next query goes to. 'standard' keeps the original
   * behaviour on port 8002; 'agentic' targets the LangGraph service on 8005.
   */
  activeMode = signal<AgentMode>('standard');

  /** Zoneless-safe derived label for the loading banner. */
  loadingLabel = computed(() =>
    this.activeMode() === 'agentic'
      ? 'Routing intent, retrieving and verifying grounding...'
      : 'Searching Knowledge Base & Generating Answer...'
  );

  private ragService = inject(RagService);

  setMode(mode: AgentMode): void {
    if (this.isLoading() || this.activeMode() === mode) return;
    this.activeMode.set(mode);
  }

  selectPrompt(promptText: string): void {
    this.searchQuery.set(promptText);
    this.onSearch();
  }

  onSearch(): void {
    const query = this.searchQuery().trim();
    if (!query || this.isLoading()) return;

    const mode = this.activeMode();

    // Add user query to conversation
    this.messages.update((msgs) => [...msgs, { role: 'user', content: query }]);
    this.searchQuery.set('');
    this.isLoading.set(true);
    this.errorMessage.set(null);

    this.ragService.sendQueryForMode(query, mode).subscribe({
      next: (response) => {
        // Every field is copied into a fresh object and pushed through
        // signal.update(), so change detection fires without Zone.js.
        this.messages.update((msgs) => [
          ...msgs,
          {
            role: 'ai',
            content: response.content,
            sources: this.uniqueSources(response.sources),
            queryLogId: response.queryLogId,
            userFeedback: response.userFeedback ?? null,
            // Trust the server's own label when present, else the mode used.
            agentMode: response.agentMode ?? mode,
            citations: response.citations ?? [],
            intent: response.intent,
            answered: response.answered,
            grounded: response.grounded,
            confidenceScore: response.confidenceScore,
            apiCallsUsed: response.apiCallsUsed,
            embeddingCallsUsed: response.embeddingCallsUsed,
            retryCount: response.retryCount,
            cached: response.cached,
            graphPath: response.graphPath ?? [],
            latencyMs: response.latencyMs,
          },
        ]);
        this.isLoading.set(false);
      },
      error: (err) => {
        console.error(`${mode} query failed:`, err);
        this.errorMessage.set(
          err?.message || 'Failed to get a response. Please try again.'
        );
        this.isLoading.set(false);
      },
    });
  }

  /** Percentage string for the confidence chip, or null when not applicable. */
  confidenceLabel(msg: ChatMessage): string | null {
    if (msg.agentMode !== 'agentic' || !msg.answered || msg.confidenceScore == null) {
      return null;
    }
    return `${Math.round(msg.confidenceScore * 100)}%`;
  }

  /**
   * Verification status for the agentic header chip.
   *
   * Kept out of the template because the four states are not mutually
   * derivable from one flag: a cache hit reports 0 API calls but is a real
   * verified answer, and a "not found" reply is faithful to its context yet
   * must not be badged as a grounded answer.
   */
  agentStatus(msg: ChatMessage): 'greeting' | 'grounded' | 'unverified' | 'no-match' | null {
    if (msg.agentMode !== 'agentic') return null;
    if (msg.intent === 'greeting') return 'greeting';
    if (!msg.answered) return 'no-match';
    return msg.grounded ? 'grounded' : 'unverified';
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