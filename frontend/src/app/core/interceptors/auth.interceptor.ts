import { inject } from '@angular/core';
import { HttpRequest, HttpHandlerFn, HttpEvent } from '@angular/common/http';
import { AuthService } from '@auth0/auth0-angular';
import { Observable, switchMap, catchError, timeout, of } from 'rxjs';
import { environment } from '../../../environments/environment';

/**
 * Functional HTTP interceptor that attaches the Auth0 Bearer token
 * to all outgoing requests targeting the backend API.
 * Includes timeout and error handling so requests never hang the UI.
 */
export function authInterceptor(
  req: HttpRequest<unknown>,
  next: HttpHandlerFn
): Observable<HttpEvent<unknown>> {
  // Only attach token to requests going to our own API
  if (!req.url.startsWith(environment.apiUrl)) {
    return next(req);
  }

  const auth0 = inject(AuthService);

  return auth0.getAccessTokenSilently().pipe(
    timeout(3000), // Don't hang for more than 3 seconds
    switchMap((token) => {
      const cloned = req.clone({
        setHeaders: {
          Authorization: `Bearer ${token}`,
        },
      });
      return next(cloned);
    }),
    catchError((err) => {
      console.warn('[AuthInterceptor] Silent token retrieval skipped or timed out:', err?.message || err);
      // Forward the request anyway so the backend responds and stops UI spinners
      return next(req);
    })
  );
}
