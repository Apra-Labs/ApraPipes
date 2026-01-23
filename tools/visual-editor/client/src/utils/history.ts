/**
 * History Manager - Undo/Redo utility using memento pattern
 *
 * Provides undo/redo functionality by maintaining snapshots of state.
 * Uses structural sharing where possible to minimize memory usage.
 */

/**
 * Configuration for the history manager
 */
export interface HistoryConfig {
  /** Maximum number of history states to keep (default: 50) */
  maxSize: number;
  /** Key for localStorage persistence (optional) */
  storageKey?: string;
}

/**
 * History entry containing a snapshot and optional metadata
 */
interface HistoryEntry<T> {
  /** The state snapshot */
  state: T;
  /** Timestamp when the snapshot was taken */
  timestamp: number;
  /** Optional description of the action that led to this state */
  action?: string;
}

/**
 * History manager class using memento pattern
 */
export class HistoryManager<T> {
  private past: HistoryEntry<T>[] = [];
  private future: HistoryEntry<T>[] = [];
  private config: HistoryConfig;
  private isTransacting = false;
  private transactionStart: T | null = null;

  constructor(config: Partial<HistoryConfig> = {}) {
    this.config = {
      maxSize: 50,
      ...config,
    };

    // Load from localStorage if configured
    if (this.config.storageKey) {
      this.loadFromStorage();
    }
  }

  /**
   * Push a new state onto the history stack
   * Clears the redo stack (future states)
   */
  push(state: T, action?: string): void {
    // If we're in a transaction, don't push intermediate states
    if (this.isTransacting) {
      return;
    }

    const entry: HistoryEntry<T> = {
      state: this.clone(state),
      timestamp: Date.now(),
      action,
    };

    this.past.push(entry);

    // Clear future (redo stack) when new action is taken
    this.future = [];

    // Enforce max size
    while (this.past.length > this.config.maxSize) {
      this.past.shift();
    }

    // Persist if configured
    if (this.config.storageKey) {
      this.saveToStorage();
    }
  }

  /**
   * Undo the last action
   * Returns the previous state or null if no more history
   */
  undo(): T | null {
    if (!this.canUndo()) {
      return null;
    }

    const current = this.past.pop()!;
    this.future.unshift(current);

    const previousState = this.past.length > 0
      ? this.past[this.past.length - 1].state
      : null;

    if (this.config.storageKey) {
      this.saveToStorage();
    }

    return previousState ? this.clone(previousState) : null;
  }

  /**
   * Redo the last undone action
   * Returns the next state or null if no more redo history
   */
  redo(): T | null {
    if (!this.canRedo()) {
      return null;
    }

    const next = this.future.shift()!;
    this.past.push(next);

    if (this.config.storageKey) {
      this.saveToStorage();
    }

    return this.clone(next.state);
  }

  /**
   * Check if undo is available
   */
  canUndo(): boolean {
    // Need at least 2 states: current + one to go back to
    return this.past.length > 1;
  }

  /**
   * Check if redo is available
   */
  canRedo(): boolean {
    return this.future.length > 0;
  }

  /**
   * Get the current state without modifying history
   */
  getCurrentState(): T | null {
    if (this.past.length === 0) {
      return null;
    }
    return this.clone(this.past[this.past.length - 1].state);
  }

  /**
   * Clear all history
   */
  clear(): void {
    this.past = [];
    this.future = [];

    if (this.config.storageKey) {
      localStorage.removeItem(this.config.storageKey);
    }
  }

  /**
   * Begin a transaction - multiple changes are grouped as one undo action
   * Call endTransaction() to commit
   */
  beginTransaction(currentState: T): void {
    if (this.isTransacting) {
      return;
    }
    this.isTransacting = true;
    this.transactionStart = this.clone(currentState);
  }

  /**
   * End a transaction and push the final state
   */
  endTransaction(finalState: T, action?: string): void {
    if (!this.isTransacting) {
      return;
    }
    this.isTransacting = false;

    // Push the state before the transaction started
    if (this.transactionStart !== null) {
      // Only push if there's an actual change
      if (JSON.stringify(this.transactionStart) !== JSON.stringify(finalState)) {
        // Push the pre-transaction state first (if not already in history)
        if (this.past.length === 0) {
          this.past.push({
            state: this.transactionStart,
            timestamp: Date.now(),
          });
        }
        // Then push the final state
        this.past.push({
          state: this.clone(finalState),
          timestamp: Date.now(),
          action,
        });
        this.future = [];

        // Enforce max size
        while (this.past.length > this.config.maxSize) {
          this.past.shift();
        }

        if (this.config.storageKey) {
          this.saveToStorage();
        }
      }
    }

    this.transactionStart = null;
  }

  /**
   * Cancel a transaction without saving
   */
  cancelTransaction(): void {
    this.isTransacting = false;
    this.transactionStart = null;
  }

  /**
   * Get the number of undo steps available
   */
  getUndoCount(): number {
    return Math.max(0, this.past.length - 1);
  }

  /**
   * Get the number of redo steps available
   */
  getRedoCount(): number {
    return this.future.length;
  }

  /**
   * Get the action description for the last undo-able state
   */
  getLastAction(): string | undefined {
    if (this.past.length < 2) {
      return undefined;
    }
    return this.past[this.past.length - 1].action;
  }

  /**
   * Deep clone an object (simple implementation using JSON)
   */
  private clone(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
  }

  /**
   * Save history to localStorage
   */
  private saveToStorage(): void {
    if (!this.config.storageKey) {
      return;
    }

    try {
      const data = {
        past: this.past,
        future: this.future,
      };
      localStorage.setItem(this.config.storageKey, JSON.stringify(data));
    } catch (error) {
      // Storage might be full or unavailable
      // eslint-disable-next-line no-console
      console.warn('Failed to save history to localStorage:', error);
    }
  }

  /**
   * Load history from localStorage
   */
  private loadFromStorage(): void {
    if (!this.config.storageKey) {
      return;
    }

    try {
      const data = localStorage.getItem(this.config.storageKey);
      if (data) {
        const parsed = JSON.parse(data);
        this.past = parsed.past || [];
        this.future = parsed.future || [];
      }
    } catch (error) {
      // Invalid data in storage
      // eslint-disable-next-line no-console
      console.warn('Failed to load history from localStorage:', error);
    }
  }
}

/**
 * Create a history manager instance
 */
export function createHistoryManager<T>(config?: Partial<HistoryConfig>): HistoryManager<T> {
  return new HistoryManager<T>(config);
}

export default HistoryManager;
