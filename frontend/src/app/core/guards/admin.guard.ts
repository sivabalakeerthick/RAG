import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '@auth0/auth0-angular';
import { firstValueFrom, filter } from 'rxjs';
import { AuthService as AppAuthService } from '../services/auth.service';

export const adminGuard: CanActivateFn = async () => {
  const auth0 = inject(AuthService);
  const appAuth = inject(AppAuthService);
  const router = inject(Router);

  // Wait for Auth0 to finish restoring the session using a proper Observable,
  // instead of a fragile polling loop.
  await firstValueFrom(auth0.isLoading$.pipe(filter((loading) => !loading)));

  const isAuthenticated = await firstValueFrom(auth0.isAuthenticated$);

  if (!isAuthenticated) {
    return router.createUrlTree(['/login']);
  }

  if (!appAuth.hasRole('Admin')) {
    return router.createUrlTree(['/unauthorized']);
  }

  return true;
};
