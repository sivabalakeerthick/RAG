# Troubleshooting

## Spinners that never stop / values stuck at their initial state

**This frontend runs without `zone.js` (zoneless change detection).** It is the
single most likely cause of any "the data arrived but the UI never updated" bug
in this app. Read this before looking at the backend.

### Symptoms

- A loading spinner appears on click and then spins forever, even though the
  operation succeeded (network tab shows `200`, the record is in the database).
- Dashboard metrics stay at `0`.
- A list stays empty, or "Loading…" never clears.
- A toast notification appears but never auto-dismisses.
- The same failure happens on features that never call the backend at all
  (e.g. the EmailJS contact form) — this is the giveaway that it is **not** a
  backend problem.

### Cause

With zoneless change detection, Angular re-renders only when something
explicitly notifies it. Template event bindings like `(click)` **do** notify,
which is why the spinner appears in the first place. But assigning a plain
class field inside an async callback notifies nothing:

```ts
// BROKEN in this app — view never re-renders
isLoading = false;

load() {
  this.isLoading = true;              // (click) notified Angular → spinner shows
  this.svc.get().subscribe(docs => {
    this.docs = docs;
    this.isLoading = false;           // nothing notifies → spinner spins forever
  });
}
```

`setTimeout`, `setInterval`, and promise `.then()` are affected exactly the
same way, because zone.js is not present to patch them.

### Fix

Any component state written from an async callback must be a `signal`. Setting
a signal notifies Angular:

```ts
// CORRECT
isLoading = signal(false);
docs = signal<DocumentFile[]>([]);

load() {
  this.isLoading.set(true);
  this.svc.get().subscribe(d => {
    this.docs.set(d);
    this.isLoading.set(false);        // signal write → view re-renders
  });
}
```

And read it as a function call in the template:

```html
@if (isLoading()) { <spinner /> }
@for (doc of docs(); track doc.id) { ... }
```

### Rules of thumb

| State written from… | Needs a signal? |
|---|---|
| An async callback (`subscribe`, `.then`, `setTimeout`) | **Yes** |
| A template event (`(click)`, `(ngModelChange)`) | No — the event notifies |
| A `@HostListener` | No — Angular registers it |
| An RxJS stream you can convert | Prefer `toSignal()` |

Derived values should be `computed()`, not getters, so they track their
signal dependencies.

`[(ngModel)]` cannot two-way bind to a signal. Split it:

```html
<input [ngModel]="query()" (ngModelChange)="query.set($event)" />
```

### Guardrail

[app.config.ts](frontend/src/app/app.config.ts) calls
`provideZonelessChangeDetection()` so this is explicit rather than an accident
of `zone.js` being absent from `package.json`.

If you would rather go back to zone-based change detection, install `zone.js`,
add `"polyfills": ["zone.js"]` to the build options in
[angular.json](frontend/angular.json), and remove that provider. Plain-field
mutation then works everywhere again.

---

## Backend: services and prerequisites

PostgreSQL must be running before you start the backend.

```bash
# Docker
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=root -e POSTGRES_DB=cognidoc \
  --name cognidoc-postgres postgres:16
```

Verify configuration, database connectivity, and ChromaDB before launching:

```bash
cd backend
python verify_setup.py
```

Start all five services:

```bash
cd backend
python run_all.py
```

| Service | Port |
|---|---|
| API Gateway | 8000 |
| Document Service | 8001 |
| RAG Service | 8002 |
| Judge Service | 8003 |
| Metrics Service | 8004 |

Health check any of them at `http://localhost:<port>/health`.

## Storage layout

- **PostgreSQL** — `documents` and `chunks` tables. `chunks.chroma_id` is the
  foreign reference into the vector store.
- **ChromaDB** — persisted on disk at `backend/chroma_data/`
  (`CHROMA_PERSIST_PATH`).
- **Uploaded files** — `backend/uploads/`, prefixed with a UUID.

Upload and delete are ordered so the two stores stay consistent: on upload,
vectors are written before the Postgres transaction is allowed to commit, and a
failure rolls back both. On delete, ChromaDB vectors are removed first, and the
Postgres row is left in place if that fails.
