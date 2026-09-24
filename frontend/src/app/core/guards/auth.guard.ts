import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '@auth0/auth0-angular';
import { firstValueFrom, filter } from 'rxjs';

export const authGuard: CanActivateFn = async () => {
  const auth0 = inject(AuthService);
  const router = inject(Router);

  // Wait for Auth0 to finish restoring the session using a proper Observable,
  // instead of a fragile polling loop.
  await firstValueFrom(auth0.isLoading$.pipe(filter((loading) => !loading)));

  const isAuthenticated = await firstValueFrom(auth0.isAuthenticated$);

  if (!isAuthenticated) {
    return router.createUrlTree(['/login']);
  }

  return true;
};
