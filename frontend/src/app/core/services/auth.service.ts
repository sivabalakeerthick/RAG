import { Injectable, inject } from '@angular/core';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { toSignal } from '@angular/core/rxjs-interop';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { LoggerService } from './logger.service';
import { ToastService } from './toast.service';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private auth0 = inject(Auth0Service);
  private http = inject(HttpClient);
  private logger = inject(LoggerService);
  private toastService = inject(ToastService);

  private auth0Domain = environment.auth0.domain;
  private clientId = environment.auth0.clientId;

  readonly isAuthenticated = toSignal(this.auth0.isAuthenticated$, { initialValue: false });
  readonly user = toSignal(this.auth0.user$, { initialValue: null });
  readonly isLoading = toSignal(this.auth0.isLoading$, { initialValue: true });

  constructor() {
    this.auth0.error$.subscribe((err) => {
      if (err) {
        this.logger.error('Global Auth0 error occurred:', err);
      }
    });
  }

  login(): void {
    this.logger.info('Initiating Auth0 login redirect', {
      domain: this.auth0Domain,
      clientId: this.clientId,
    });
    
    this.auth0.loginWithRedirect({
      appState: { target: '/home' },
    }).subscribe({
      next: () => this.logger.debug('Auth0 loginWithRedirect initiated'),
      error: (err) => {
        this.logger.error('Auth0 loginWithRedirect failed:', err);
        this.toastService.error(
          'Unable to sign in right now. Please verify your connection and try again.',
          'Authentication Error'
        );
      },
    });
  }

  logout(): void {
    this.logger.info('User initiated logout');
    sessionStorage.clear();
    localStorage.clear();

    this.auth0.logout({
      logoutParams: {
        returnTo: window.location.origin,
      },
    }).subscribe({
      error: (err) => this.logger.error('Auth0 logout error:', err),
    });
  }

  getAccessToken() {
    return this.auth0.getAccessTokenSilently();
  }

  getUserRoles(user?: any): string[] {
    const u = user ?? this.user();
    if (!u) return [];

    const raw =
      u['https://angular-app.com/roles'] ??
      u['https://cognidoc-api/roles'] ??
      u['https://cognidoc.com/roles'] ??
      u['roles'] ??
      u['user_metadata']?.roles ??
      u['app_metadata']?.roles ??
      [];

    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'string') return [raw];
    return [];
  }

  hasRole(role: string, targetUser?: any): boolean {
    const roles = this.getUserRoles(targetUser);
    return roles.some((r) => r.toLowerCase() === role.toLowerCase());
  }

  hasAnyRole(roles: string[], targetUser?: any): boolean {
    const userRoles = this.getUserRoles(targetUser);
    const lowerRoles = roles.map((r) => r.toLowerCase());
    return userRoles.some((r) => lowerRoles.includes(r.toLowerCase()));
  }

  // Auth0 Change Password Request
  sendPasswordResetEmail(email: string): Observable<any> {
    const url = `https://${this.auth0Domain}/dbconnections/change_password`;
    const body = {
      client_id: this.clientId,
      email: email,
      connection: 'Username-Password-Authentication',
    };

    return this.http.post(url, body, { responseType: 'text' });
  }
}