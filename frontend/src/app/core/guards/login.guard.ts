import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from '@auth0/auth0-angular';
import { filter, firstValueFrom } from 'rxjs';

/**
 * Prevents authenticated users or in-flight Auth0 login callbacks
 * from showing the Login component. Redirects authenticated users to /home.
 */
export const loginGuard: CanActivateFn = async () => {
  const auth0 = inject(AuthService);
  const router = inject(Router);

  // Wait for Auth0 to finish restoring session or exchanging redirect callback code
  await firstValueFrom(auth0.isLoading$.pipe(filter((loading) => !loading)));

  const isAuthenticated = await firstValueFrom(auth0.isAuthenticated$);

  if (isAuthenticated) {
    return router.createUrlTree(['/home']);
  }

  return true;
};
