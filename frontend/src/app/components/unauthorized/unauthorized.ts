import { Component } from '@angular/core';
import { Router,  } from '@angular/router';
import { MatCardModule } from '@angular/material/card';

@Component({
  selector: 'app-unauthorized',
  standalone: true,
  imports: [MatCardModule],
  templateUrl: './unauthorized.html',
  styleUrl: './unauthorized.scss'
})
export class Unauthorized {
  constructor(private router: Router) {}

  goHome(): void {
    this.router.navigate(['/home']);
  }
}