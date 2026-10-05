import { Component, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatToolbarModule } from '@angular/material/toolbar';
import { AuthService } from '../../core/services/auth.service';
import { LoggerService } from '../../core/services/logger.service';
import { ToastService } from '../../core/services/toast.service';

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

  constructor(
    public authService: AuthService,
    private logger: LoggerService,
    private toastService: ToastService
  ) {}

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
      this.toastService.warning('Unable to find user email address.', 'Password Reset');
      return;
    }

    this.isResetSentModalOpen = true;
    this.logger.info('Requesting password reset email via Auth0', { email: userEmail });

    this.authService.sendPasswordResetEmail(userEmail).subscribe({
      next: () => this.logger.info('Password reset email sent via Auth0'),
      error: (err) => this.logger.warn('Auth0 password reset response:', err)
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