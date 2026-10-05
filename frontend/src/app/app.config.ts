import {
  ApplicationConfig,
  ErrorHandler,
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
} from '@angular/core';
import { provideRouter, withEnabledBlockingInitialNavigation } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideAuth0 } from '@auth0/auth0-angular';

import { routes } from './app.routes';
import { environment } from '../environments/environment';
import { authInterceptor } from './core/interceptors/auth.interceptor';
import { errorInterceptor } from './core/interceptors/error.interceptor';
import { GlobalErrorHandler } from './core/services/global-error-handler';

export const appConfig: ApplicationConfig = {
  providers: [
    // This app runs without zone.js. Component state that changes in async
    // callbacks must be a signal, or the view will not re-render.
    provideZonelessChangeDetection(),
    provideBrowserGlobalErrorListeners(),
    provideRouter(routes, withEnabledBlockingInitialNavigation()),
    provideHttpClient(withInterceptors([authInterceptor, errorInterceptor])),
    { provide: ErrorHandler, useClass: GlobalErrorHandler },
    provideAuth0({
      domain: environment.auth0.domain,
      clientId: environment.auth0.clientId,
      authorizationParams: {
        redirect_uri: window.location.origin,
        audience: environment.auth0.audience,
      },
      useRefreshTokens: true,
      cacheLocation: 'localstorage',
    }),
  ],
};
