import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { Dashboard } from './dashboard';
import { DocumentService } from '../../core/services/document.service';

describe('Dashboard', () => {
  let component: Dashboard;
  let fixture: ComponentFixture<Dashboard>;

  const mockDocService = {
    cachedDocuments: signal([]),
    cachedMetrics: signal(null),
    cachedIndexHealth: signal(null),
    getDocuments: () => of([]),
    getMetrics: () => of({
      indexedDocuments: 0,
      processingDocuments: 0,
      vectorChunks: 0,
      monthlyQueries: 0,
      groundingRate: '0%',
      avgLatencyMs: 0,
      precision: '0%',
      recall: '0%',
      f1Score: '0%',
      tp: 0,
      tn: 0,
      fp: 0,
      fn: 0,
      userFeedbackCount: 0,
      positiveFeedbackCount: 0,
      negativeFeedbackCount: 0,
    }),
    getIndexHealth: () => of({ chunkRows: 0, vectors: 0, inSync: true }),
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Dashboard],
      providers: [{ provide: DocumentService, useValue: mockDocService }],
    }).compileComponents();

    fixture = TestBed.createComponent(Dashboard);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
