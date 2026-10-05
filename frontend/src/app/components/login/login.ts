import { Component, inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { AuthService } from '../../core/services/auth.service';
import { LoggerService } from '../../core/services/logger.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    MatButtonModule,
    MatCardModule
  ],
  templateUrl: './login.html',
  styleUrl: './login.scss'
})
export class Login {

  protected authService = inject(AuthService);
  private logger = inject(LoggerService);

  login(): void {
    this.logger.info('User clicked Sign In / Sign Up button');
    this.authService.login();
  }
}