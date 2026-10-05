import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { signal } from '@angular/core';
import { App } from './app';
import { AuthService } from './core/services/auth.service';

describe('App', () => {
  const mockAuthService = {
    isLoading: signal(false),
    isAuthenticated: signal(true),
    user: signal({ name: 'Test User', email: 'test@example.com' }),
    hasRole: () => true,
    hasAnyRole: () => true,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideRouter([]),
        { provide: AuthService, useValue: mockAuthService },
      ],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });
});
