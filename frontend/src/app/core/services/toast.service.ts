import { Injectable, signal } from '@angular/core';

export type ToastType = 'error' | 'warning' | 'info' | 'success';

export interface ToastMessage {
  id: string;
  title: string;
  message: string;
  type: ToastType;
  duration: number;
  timestamp: number;
}

@Injectable({
  providedIn: 'root',
})
export class ToastService {
  readonly toasts = signal<ToastMessage[]>([]);
  private recentMessages = new Map<string, number>();

  show(
    message: string,
    type: ToastType = 'error',
    title?: string,
    duration = 5000
  ): string {
    const trimmed = (message || '').trim();
    if (!trimmed) return '';

    const defaultTitle =
      title ||
      (type === 'error'
        ? 'Error'
        : type === 'warning'
        ? 'Warning'
        : type === 'success'
        ? 'Success'
        : 'Notice');

    const key = `${type}:${defaultTitle}:${trimmed}`;
    const now = Date.now();
    const lastTime = this.recentMessages.get(key);

    // Suppress duplicate toasts fired within 4 seconds (e.g. concurrent HTTP failures)
    if (lastTime && now - lastTime < 4000) {
      return '';
    }
    this.recentMessages.set(key, now);

    // Also check if an active toast already has the exact same message
    if (this.toasts().some((t) => t.message === trimmed && t.type === type)) {
      return '';
    }

    const id = `toast-${now}-${Math.random().toString(36).substring(2, 7)}`;

    const toast: ToastMessage = {
      id,
      title: defaultTitle,
      message: trimmed,
      type,
      duration,
      timestamp: now,
    };

    // Keep at most 2 simultaneous toasts to maintain a clean, uncluttered UI
    this.toasts.update((current) => [toast, ...current.slice(0, 1)]);

    if (duration > 0) {
      setTimeout(() => {
        this.dismiss(id);
      }, duration);
    }

    return id;
  }

  error(message: string, title?: string, duration = 6000): string {
    return this.show(message, 'error', title || 'Request Failed', duration);
  }

  warning(message: string, title?: string, duration = 6000): string {
    return this.show(message, 'warning', title || 'Warning', duration);
  }

  info(message: string, title?: string, duration = 5000): string {
    return this.show(message, 'info', title || 'Information', duration);
  }

  success(message: string, title?: string, duration = 4000): string {
    return this.show(message, 'success', title || 'Success', duration);
  }

  dismiss(id: string): void {
    this.toasts.update((current) => current.filter((t) => t.id !== id));
  }

  clear(): void {
    this.toasts.set([]);
  }
}
