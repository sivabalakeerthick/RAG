import { Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterModule, MatToolbarModule],
  templateUrl: './nav-bar.html',
  styleUrl: './nav-bar.scss'
})
export class Navbar {
  isProfileCardOpen = false;
  isLogoutConfirmOpen = false;
  isResetConfirmOpen = false;
  isResetSentModalOpen = false; // Flag controls the confirmation card

  constructor(public authService: AuthService) {}

  toggleProfileCard(event: Event): void {
    event.stopPropagation();
    this.isProfileCardOpen = !this.isProfileCardOpen;
  }

  onResetPassword(): void {
    this.isProfileCardOpen = false;
    this.isResetConfirmOpen = true;
  }

  cancelResetPassword(): void {
    this.isResetConfirmOpen = false;
  }

  confirmResetPassword(): void {
    this.isResetConfirmOpen = false;
    const currentUser = this.authService.user();
    const userEmail = currentUser?.email;

    if (!userEmail) {
      alert('Unable to find user email address.');
      return;
    }

    this.isResetSentModalOpen = true;

    this.authService.sendPasswordResetEmail(userEmail).subscribe({
      next: () => console.log('Reset email sent via Auth0'),
      error: (err) => console.log('Auth0 API notice:', err)
    });
  }

  openLogoutConfirmation(): void {
    this.isProfileCardOpen = false;
    this.isLogoutConfirmOpen = true;
  }

  cancelLogout(): void {
    this.isLogoutConfirmOpen = false;
  }

  confirmLogout(): void {
    this.isLogoutConfirmOpen = false;
    this.authService.logout();
  }

  @HostListener('document:click')
  onDocumentClick(): void {
    this.isProfileCardOpen = false;
  }
}