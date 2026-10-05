import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Login } from './login';
import { AuthService } from '../../core/services/auth.service';

describe('Login', () => {
  let component: Login;
  let fixture: ComponentFixture<Login>;

  const mockAuthService = {
    isLoading: signal(false),
    isAuthenticated: signal(false),
    user: signal(null),
    login: () => {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Login],
      providers: [{ provide: AuthService, useValue: mockAuthService }],
    }).compileComponents();

    fixture = TestBed.createComponent(Login);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
