import { Component, inject, signal } from '@angular/core';
import { Router, RouterOutlet } from '@angular/router';
import { Navbar } from './shared/nav-bar/nav-bar';
import { Footer } from './shared/footer/footer';
import { AuthService } from './core/services/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet,Navbar,Footer],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected router = inject(Router);
  protected authService = inject(AuthService);
  protected readonly title = signal('frontend');
  hiddenNavbarRoutes = ['/login', '/unauthorized', '/page-not-found'];

  constructor() {
    // Detects BFCache restoration on Browser Back button and cached DOM
    window.addEventListener('pageshow', (event) => {
      console.log('persisted:', event.persisted);
      if (event.persisted) {
        window.location.reload();
      }
    });
  }

  get showNavbar(): boolean {
    return !this.hiddenNavbarRoutes.some((route) => this.router.url.startsWith(route));
  }
}
