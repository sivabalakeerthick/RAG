import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '@auth0/auth0-angular';
import { firstValueFrom, filter, take, timeout, of } from 'rxjs';
import { AuthService as AppAuthService } from '../services/auth.service';

export const adminGuard: CanActivateFn = async () => {
  const auth0 = inject(AuthService);
  const appAuth = inject(AppAuthService);
  const router = inject(Router);

  // 1. Wait for Auth0 to finish restoring the session
  await firstValueFrom(auth0.isLoading$.pipe(filter((loading) => !loading)));

  // 2. Check authentication status
  const isAuthenticated = await firstValueFrom(auth0.isAuthenticated$);
  if (!isAuthenticated) {
    return router.createUrlTree(['/login']);
  }

  // 3. Wait for user profile to be populated on page refresh
  let user = appAuth.user();
  if (!user) {
    try {
      user = await firstValueFrom(
        auth0.user$.pipe(
          filter((u): u is NonNullable<typeof u> => !!u),
          take(1),
          timeout({ each: 4000, with: () => of(null) })
        )
      );
    } catch {
      user = null;
    }
  }

  // 4. Verify Admin role against the resolved user
  if (!user || !appAuth.hasRole('Admin', user)) {
    return router.createUrlTree(['/unauthorized']);
  }

  return true;
};
