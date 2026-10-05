import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { signal } from '@angular/core';
import { Navbar } from './nav-bar';
import { AuthService } from '../../core/services/auth.service';

describe('NavBar', () => {
  let component: Navbar;
  let fixture: ComponentFixture<Navbar>;

  const mockAuthService = {
    isLoading: signal(false),
    isAuthenticated: signal(true),
    user: signal({ name: 'Admin User', email: 'admin@cognidoc.com' }),
    hasRole: (role: string) => role.toLowerCase() === 'admin',
    hasAnyRole: (roles: string[]) => roles.some((r) => r.toLowerCase() === 'admin'),
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Navbar],
      providers: [
        provideRouter([]),
        { provide: AuthService, useValue: mockAuthService },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(Navbar);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
