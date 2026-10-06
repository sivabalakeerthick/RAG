import { Component, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { AgentMode, ChatMessage, SourceChunk } from '../../models/chat.model';
import { RagService } from '../../core/services/rag.service';
import { LoggerService } from '../../core/services/logger.service';
import { formatUserError } from '../../core/utils/error-formatter';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  private ragService = inject(RagService);
  private logger = inject(LoggerService);
  private sanitizer = inject(DomSanitizer);

  searchQuery = signal('');
  isLoading = signal(false);
  errorMessage = signal<string | null>(null);
  lastFailedQuery = signal<string | null>(null);
  copiedId = signal<string | null>(null);

  // ── Connected to singleton RagService (persists across component navigation) ──
  messages = this.ragService.messages;
  sessionId = this.ragService.sessionId;
  activeMode = this.ragService.activeMode;

  /** Zoneless-safe derived label for the loading banner. */
  loadingLabel = computed(() =>
    this.activeMode() === 'agentic'
      ? 'Routing intent, retrieving and verifying grounding...'
      : 'Searching Knowledge Base & Generating Answer...'
  );

  clearChat(): void {
    if (this.isLoading()) return;
    this.ragService.clearChat();
    this.errorMessage.set(null);
    this.lastFailedQuery.set(null);
    this.searchQuery.set('');
    this.logger.info('Conversation cleared');
  }

  resetChat(): void {
    this.clearChat();
  }

  newChat(): void {
    this.clearChat();
  }

  setMode(mode: AgentMode): void {
    if (this.isLoading() || this.activeMode() === mode) return;
    this.logger.info(`Switching query mode to "${mode}"`);
    this.ragService.setMode(mode);
  }

  selectPrompt(promptText: string): void {
    this.searchQuery.set(promptText);
    this.onSearch();
  }

  onSearch(customQuery?: string): void {
    const query = (customQuery ?? this.searchQuery()).trim();
    if (!query || this.isLoading()) return;

    const mode = this.activeMode();

    // If not a retry, push new user question to conversation
    if (!customQuery) {
      this.ragService.addMessage({ role: 'user', content: query });
      this.searchQuery.set('');
    }

    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.logger.info(`Dispatching user query [mode=${mode}]`, { queryLength: query.length });

    this.ragService.sendQueryForMode(query, mode, this.sessionId()).subscribe({
      next: (response) => {
        this.logger.info(`Received response for [mode=${mode}]`, {
          latencyMs: response.latencyMs,
          answered: response.answered,
          grounded: response.grounded,
          sourcesCount: response.sources?.length ?? 0,
        });

        this.lastFailedQuery.set(null);

        // Add AI response to persistent conversation
        this.ragService.addMessage({
          role: 'ai',
          content: this.cleanContent(response.content || ''),
          sources: this.uniqueSources(response.sources),
          queryLogId: response.queryLogId,
          userFeedback: response.userFeedback ?? null,
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
          toolCalls: response.toolCalls ?? [],
          sessionId: response.sessionId,
          latencyMs: response.latencyMs,
        });
        this.isLoading.set(false);
      },
      error: (err) => {
        this.logger.error(`${mode} query failed:`, err);
        const formatted = formatUserError(
          err,
          mode === 'agentic'
            ? 'The assistant could not complete your request. Please try again.'
            : 'Failed to process your search request. Please try again.'
        );
        this.lastFailedQuery.set(query);
        this.errorMessage.set(formatted.message);
        this.isLoading.set(false);
      },
    });
  }

  retryQuery(): void {
    const query = this.lastFailedQuery();
    if (query) {
      this.onSearch(query);
    }
  }

  copyText(text: string, id: string): void {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      this.copiedId.set(id);
      setTimeout(() => {
        if (this.copiedId() === id) {
          this.copiedId.set(null);
        }
      }, 2000);
    }).catch((err) => {
      this.logger.error('Failed to copy to clipboard:', err);
    });
  }

  confidenceLabel(msg: ChatMessage): string {
    if (msg.confidenceScore === undefined || msg.confidenceScore === null) {
      return '';
    }
    return `${Math.round(msg.confidenceScore * 100)}%`;
  }

  agentStatus(msg: ChatMessage): 'greeting' | 'grounded' | 'unverified' | 'no-match' | null {
    if (msg.agentMode !== 'agentic') return null;
    if (msg.intent === 'greeting') return 'greeting';
    if (!msg.answered) return 'no-match';
    return msg.grounded ? 'grounded' : 'unverified';
  }

  submitFeedback(msg: ChatMessage, feedback: 'thumbs_up' | 'thumbs_down'): void {
    if (!msg.queryLogId || msg.userFeedback === feedback || msg.isFeedbackSubmitting) return;

    this.logger.info(`Submitting feedback "${feedback}" for query log ${msg.queryLogId}`);
    msg.isFeedbackSubmitting = true;

    this.ragService.submitFeedback(msg.queryLogId, feedback).subscribe({
      next: () => {
        this.logger.info(`Feedback recorded for query log ${msg.queryLogId}`);
        this.ragService.updateMessageFeedback(msg.queryLogId!, feedback);
      },
      error: (err) => {
        this.logger.error('Failed to submit feedback:', err);
        msg.isFeedbackSubmitting = false;
      },
    });
  }

  private uniqueSources(sources?: SourceChunk[]): SourceChunk[] {
    if (!sources) return [];

    const seen = new Set<string>();
    return sources.filter((s) => {
      if (seen.has(s.fileName)) return false;
      seen.add(s.fileName);
      return true;
    });
  }

  formatMarkdown(content: string): SafeHtml {
    if (!content) return '';

    // Escape HTML special characters to prevent XSS
    let html = content
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Markdown Headers (###, ##, #)
    html = html.replace(/^###[ \t]+(.*$)/gim, '<h4 class="md-h4">$1</h4>');
    html = html.replace(/^##[ \t]+(.*$)/gim, '<h3 class="md-h3">$1</h3>');
    html = html.replace(/^#[ \t]+(.*$)/gim, '<h2 class="md-h2">$1</h2>');

    // Bold + Italic, Bold, Italic
    html = html.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');

    // Bullet points (* Item or - Item)
    html = html.replace(/^[\*\-][ \t]+(.*$)/gim, '<div class="md-bullet"><span class="bullet-dot">•</span><span class="bullet-text">$1</span></div>');

    // Numbered lists (1. Item)
    html = html.replace(/^(\d+)\.[ \t]+(.*$)/gim, '<div class="md-num-item"><span class="num-badge">$1.</span><span class="num-text">$2</span></div>');

    // Paragraph breaks and newlines
    html = html.replace(/\n\n/g, '<div class="md-spacer"></div>');
    html = html.replace(/\n/g, '<br/>');

    return this.sanitizer.bypassSecurityTrustHtml(html);
  }

  private cleanContent(raw: string): string {
    if (!raw) return '';
    let text = raw.trim();
    // Catch stringified LangChain/Gemini parts e.g. [{'type': 'text', 'text': '...'}]
    if (text.startsWith('[') && (text.includes("'text':") || text.includes('"text":'))) {
      const match = text.match(/['"]text['"]\s*:\s*['"](.*?)['"](?:\s*,\s*['"]extras['"]|\s*})/s);
      if (match && match[1]) {
        text = match[1];
      }
    }
    return text
      .replace(/\[Source:\s*[^\]]+\]/gi, '')
      .replace(/[^\S\r\n]{2,}/g, ' ')  // Collapse horizontal spaces only (preserves newlines!)
      .replace(/\n{3,}/g, '\n\n')      // Normalize excessive empty lines
      .trim();
  }
}