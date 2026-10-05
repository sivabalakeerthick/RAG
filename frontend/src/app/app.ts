import { Component, computed, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter, map } from 'rxjs';
import { Navbar } from './shared/nav-bar/nav-bar';
import { Footer } from './shared/footer/footer';
import { ToastComponent } from './shared/toast/toast';
import { AuthService } from './core/services/auth.service';
import { LoggerService } from './core/services/logger.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, Navbar, Footer, ToastComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected router = inject(Router);
  protected authService = inject(AuthService);
  protected logger = inject(LoggerService);
  protected readonly title = signal('frontend');
  readonly hiddenNavbarRoutes = ['/login', '/unauthorized', '/page-not-found'];

  // Track the current URL as a reactive signal, initializing from window.location.pathname
  // to prevent frame-0 navbar flashing on public/login routes.
  private currentUrl = toSignal(
    this.router.events.pipe(
      filter((e): e is NavigationEnd => e instanceof NavigationEnd),
      map((e) => e.urlAfterRedirects)
    ),
    { initialValue: typeof window !== 'undefined' ? window.location.pathname : '/' }
  );

  readonly showNavbar = computed(() => {
    const url = this.currentUrl();
    if (!url || url === '/' || this.hiddenNavbarRoutes.some((route) => url.startsWith(route))) {
      return false;
    }
    return true;
  });

  constructor() {
    this.logger.info('Application initialized');
    // Detects BFCache restoration on Browser Back button and cached DOM
    window.addEventListener('pageshow', (event) => {
      this.logger.debug('pageshow persisted:', event.persisted);
      if (event.persisted) {
        window.location.reload();
      }
    });
  }
}
