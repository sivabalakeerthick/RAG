import { Routes } from '@angular/router';
import { authGuard } from './core/guards/auth.guard';
import { adminGuard } from './core/guards/admin.guard';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'login',
    pathMatch: 'full',
  },

  {
    path: 'login',
    loadComponent: () => import('./components/login/login').then((m) => m.Login),
  },

  {
    path: 'home',
    loadComponent: () => import('./components/home/home').then((m) => m.Home),
    canActivate: [authGuard],
  },

  {
    path: 'dashboard',
    loadComponent: () => import('./components/dashboard/dashboard').then((m) => m.Dashboard),
    canActivate: [adminGuard],
  },

  {
    path: 'about',
    loadComponent: () => import('./components/about/about').then((m) => m.About),
    canActivate: [authGuard],
  },

  {
    path: 'contact',
    loadComponent: () => import('./components/contact/contact').then((m) => m.Contact),
    canActivate: [authGuard],
  },

  {
    path: 'unauthorized',
    loadComponent: () =>
      import('./components/unauthorized/unauthorized').then((m) => m.Unauthorized),
  },

  {
    path: '**',
    redirectTo: 'page-not-found',
  },
  
  {
    path: 'page-not-found',
    loadComponent: () =>
      import('./components/page-not-found/page-not-found').then((m) => m.PageNotFound),
  },
];
