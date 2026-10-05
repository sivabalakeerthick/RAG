import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

@Injectable({
  providedIn: 'root',
})
export class LoggerService {
  private readonly isProd = Boolean((environment as Record<string, any>)['production']);

  debug(...args: any[]): void {
    if (this.isProd) return;
    this.dispatch('debug', '#8b5cf6', args);
  }

  info(...args: any[]): void {
    if (this.isProd) return;
    this.dispatch('info', '#3b82f6', args);
  }

  warn(...args: any[]): void {
    this.dispatch('warn', '#f59e0b', args);
  }

  error(...args: any[]): void {
    this.dispatch('error', '#ef4444', args);
  }

  private dispatch(level: LogLevel, badgeColor: string, rawArgs: any[]): void {
    if (rawArgs.length === 0) return;

    let context = 'App';
    let message = '';
    let extraArgs: any[] = [];

    // Support both signatures:
    // 1) logger.info('Message', ...extra)
    // 2) logger.info('Context', 'Message', ...extra)
    if (
      rawArgs.length >= 2 &&
      typeof rawArgs[0] === 'string' &&
      typeof rawArgs[1] === 'string' &&
      rawArgs[0].length < 25 &&
      !rawArgs[0].includes(' ')
    ) {
      context = rawArgs[0];
      message = rawArgs[1];
      extraArgs = rawArgs.slice(2);
    } else {
      message = typeof rawArgs[0] === 'string' ? rawArgs[0] : String(rawArgs[0]);
      extraArgs = rawArgs.slice(1);
    }

    this.print(level, context, message, badgeColor, extraArgs);
  }

  private print(
    level: LogLevel,
    context: string,
    message: string,
    badgeColor: string,
    args: any[]
  ): void {
    const time = new Date().toLocaleTimeString();
    const prefix = `%c[CogniDoc] %c[${context}]%c`;
    const style1 = `background: ${badgeColor}; color: white; padding: 2px 5px; border-radius: 3px; font-weight: bold; font-size: 11px;`;
    const style2 = `color: ${badgeColor}; font-weight: 600; font-size: 11px; margin-left: 4px;`;
    const style3 = `color: inherit; font-size: 11px;`;

    const consoleFn =
      level === 'error'
        ? console.error
        : level === 'warn'
        ? console.warn
        : level === 'info'
        ? console.info
        : console.debug;

    if (args.length > 0) {
      consoleFn(`${prefix} [${time}] ${message}`, style1, style2, style3, ...args);
    } else {
      consoleFn(`${prefix} [${time}] ${message}`, style1, style2, style3);
    }
  }
}
