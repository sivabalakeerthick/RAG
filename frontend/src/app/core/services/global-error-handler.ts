import { ErrorHandler, Injectable, inject } from '@angular/core';
import { ToastService } from './toast.service';
import { LoggerService } from './logger.service';
import { formatUserError } from '../utils/error-formatter';

@Injectable({
  providedIn: 'root',
})
export class GlobalErrorHandler implements ErrorHandler {
  private toastService = inject(ToastService);
  private logger = inject(LoggerService);

  handleError(error: unknown): void {
    // Structured error logging
    this.logger.error('Runtime', 'Unhandled client application error:', error);

    const rawMessage = error instanceof Error ? error.message : String(error ?? '');

    // Ignore benign / expected framework navigation events
    if (
      rawMessage.includes('ChunkLoadError') ||
      rawMessage.includes('Navigation cancelled') ||
      rawMessage.includes('NG0100')
    ) {
      return;
    }

    const formatted = formatUserError(
      error,
      'An unexpected interface issue occurred. Please refresh the page or try again.'
    );

    this.toastService.error(formatted.message, formatted.title, 6000);
  }
}
