import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { ToastService } from '../services/toast.service';
import { LoggerService } from '../services/logger.service';
import { formatUserError } from '../utils/error-formatter';

let lastOfflineToastTime = 0;

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const toastService = inject(ToastService);
  const logger = inject(LoggerService);

  return next(req).pipe(
    catchError((error: unknown) => {
      if (error instanceof HttpErrorResponse) {
        logger.error('HTTP', `${error.status} ${req.method} ${req.urlWithParams}`, error);

        // Suppress duplicate offline toasts for concurrent background failures
        if (error.status === 0) {
          const now = Date.now();
          if (now - lastOfflineToastTime < 8000) {
            return throwError(() => error);
          }
          lastOfflineToastTime = now;
        }

        const formatted = formatUserError(error);
        toastService.error(formatted.message, formatted.title);
      } else {
        logger.error('HTTP', `Non-HTTP request error on ${req.method} ${req.url}:`, error);
      }

      return throwError(() => error);
    })
  );
};
