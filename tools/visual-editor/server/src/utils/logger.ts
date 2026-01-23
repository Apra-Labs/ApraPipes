/**
 * Simple logger utility for server-side logging
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const currentLevel: LogLevel =
  (process.env.LOG_LEVEL as LogLevel) || 'info';

function formatTimestamp(): string {
  return new Date().toISOString();
}

function shouldLog(level: LogLevel): boolean {
  return LEVELS[level] >= LEVELS[currentLevel];
}

/**
 * Create a logger with a specific prefix
 */
export function createLogger(prefix: string) {
  return {
    debug: (message: string, ...args: unknown[]) => {
      if (shouldLog('debug')) {
        // eslint-disable-next-line no-console
        console.debug(`[${formatTimestamp()}] [DEBUG] [${prefix}] ${message}`, ...args);
      }
    },
    info: (message: string, ...args: unknown[]) => {
      if (shouldLog('info')) {
        // eslint-disable-next-line no-console
        console.info(`[${formatTimestamp()}] [INFO] [${prefix}] ${message}`, ...args);
      }
    },
    warn: (message: string, ...args: unknown[]) => {
      if (shouldLog('warn')) {
        console.warn(`[${formatTimestamp()}] [WARN] [${prefix}] ${message}`, ...args);
      }
    },
    error: (message: string, ...args: unknown[]) => {
      if (shouldLog('error')) {
        console.error(`[${formatTimestamp()}] [ERROR] [${prefix}] ${message}`, ...args);
      }
    },
  };
}

// Default logger instance
export const logger = createLogger('App');
