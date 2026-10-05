import { HttpErrorResponse } from '@angular/common/http';

export interface UserFriendlyError {
  title: string;
  message: string;
}

/**
 * Transforms any technical error, HTTP exception, or developer stack trace
 * into a polite, professional, user-friendly message suitable for UI display.
 */
export function formatUserError(error: unknown, fallbackMessage?: string): UserFriendlyError {
  let status = 0;
  let rawText = '';

  if (error instanceof HttpErrorResponse) {
    status = error.status;
    const body = error.error;
    if (body && typeof body === 'object') {
      rawText = body.message || body.detail || '';
      if (typeof rawText !== 'string') {
        rawText = JSON.stringify(rawText);
      }
    } else if (typeof body === 'string') {
      rawText = body;
    } else {
      rawText = error.message || error.statusText || '';
    }
  } else if (error instanceof Error) {
    rawText = error.message;
  } else if (typeof error === 'string') {
    rawText = error;
  } else if (error && typeof error === 'object' && 'message' in error) {
    rawText = String((error as any).message);
  }

  const lower = rawText.toLowerCase();

  // 1. Quota / Rate limit
  if (
    status === 429 ||
    lower.includes('resource_exhausted') ||
    lower.includes('rate limit') ||
    lower.includes('quota') ||
    lower.includes('too many requests')
  ) {
    return {
      title: 'Limit Reached',
      message: 'AI service request limit reached. Please wait about 30–45 seconds before trying again.',
    };
  }

  // 2. Server offline / Network failure / Gateway timeouts
  if (
    status === 0 ||
    status === 502 ||
    status === 503 ||
    status === 504 ||
    lower.includes('failed to fetch') ||
    lower.includes('network offline') ||
    lower.includes('econnrefused') ||
    lower.includes('connection refused') ||
    lower.includes('http failure response')
  ) {
    return {
      title: 'Connection Issue',
      message: 'Unable to connect to the server. Please check your network connection or verify that services are running.',
    };
  }

  // 3. Database / Storage issues
  if (
    lower.includes('operationalerror') ||
    lower.includes('psycopg2') ||
    lower.includes('chromadb') ||
    lower.includes('sqlite3') ||
    lower.includes('database') ||
    lower.includes('deadlock')
  ) {
    return {
      title: 'Database Busy',
      message: 'The knowledge base is temporarily busy. Please try again in a few moments.',
    };
  }

  // 4. Document extraction / chunking issues
  if (
    lower.includes('badzipfile') ||
    lower.includes('pdfsyntaxerror') ||
    lower.includes('filedataerror') ||
    lower.includes('no text could be extracted') ||
    lower.includes('unsupported file')
  ) {
    return {
      title: 'Document Processing Error',
      message: 'Could not read text from this document. Please make sure the file is valid and not password-protected.',
    };
  }

  // 5. Authentication / Authorization
  if (status === 401 || status === 403 || lower.includes('unauthorized') || lower.includes('forbidden')) {
    return {
      title: 'Access Restricted',
      message: 'You do not have permission or your session has expired. Please refresh the page.',
    };
  }

  // 6. Not Found
  if (status === 404 || lower.includes('not found')) {
    return {
      title: 'Not Found',
      message: 'The requested document or resource was not found.',
    };
  }

  // 7. Validation / Bad Input
  if (status === 400 || status === 422 || lower.includes('validation')) {
    return {
      title: 'Invalid Request',
      message: 'Please review your input and ensure all required fields are filled correctly.',
    };
  }

  // 8. If rawText is already clean and user-friendly (not containing code, json, tracebacks, line numbers)
  if (
    rawText &&
    rawText.length < 140 &&
    !rawText.includes('{') &&
    !rawText.includes('::') &&
    !rawText.includes('Traceback') &&
    !rawText.includes('line ') &&
    !rawText.includes('Http failure') &&
    !rawText.includes('Error:') &&
    !rawText.includes('Exception')
  ) {
    return {
      title: 'Notice',
      message: rawText,
    };
  }

  // 9. General fallback
  return {
    title: 'Service Error',
    message: fallbackMessage || 'An unexpected issue occurred while processing your request. Please try again.',
  };
}
