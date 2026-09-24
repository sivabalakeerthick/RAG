import { Injectable, inject } from '@angular/core';
import { AuthService as Auth0Service } from '@auth0/auth0-angular';
import { toSignal } from '@angular/core/rxjs-interop';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private auth0 = inject(Auth0Service);
  private http = inject(HttpClient);

  private auth0Domain = environment.auth0.domain;
  private clientId = environment.auth0.clientId;

  readonly isAuthenticated = toSignal(this.auth0.isAuthenticated$, { initialValue: false });
  readonly user = toSignal(this.auth0.user$, { initialValue: null });
  readonly isLoading = toSignal(this.auth0.isLoading$, { initialValue: true });

  constructor() {
    this.auth0.error$.subscribe((err) => {
      if (err) {
        console.error('[Auth0 Service] Global Auth0 error occurred:', err);
      }
    });
  }

  login(): void {
    console.log('[AuthService] Initiating Auth0 loginWithRedirect...');
    console.log('[AuthService] Domain:', this.auth0Domain, '| ClientId:', this.clientId);
    
    this.auth0.loginWithRedirect({
      appState: { target: '/home' },
    }).subscribe({
      next: () => console.log('[AuthService] loginWithRedirect emitted next (redirecting)'),
      error: (err) => {
        console.error('[AuthService] loginWithRedirect error:', err);
        alert('Auth0 Login Error: ' + (err?.message || JSON.stringify(err)));
      },
    });
  }

  logout(): void {
    sessionStorage.clear();
    localStorage.clear();

    this.auth0.logout({
      logoutParams: {
        returnTo: window.location.origin,
      },
    }).subscribe({
      error: (err) => console.error('Auth0 logout error:', err),
    });
  }

  getAccessToken() {
    return this.auth0.getAccessTokenSilently();
  }

  hasRole(role: string): boolean {
    const currentUser = this.user();
    const roles = (currentUser?.['https://angular-app.com/roles'] as string[]) ?? [];
    return roles.includes(role);
  }

  hasAnyRole(roles: string[]): boolean {
    const currentUser = this.user();
    const userRoles = (currentUser?.['https://angular-app.com/roles'] as string[]) ?? [];
    return roles.some((role) => userRoles.includes(role));
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