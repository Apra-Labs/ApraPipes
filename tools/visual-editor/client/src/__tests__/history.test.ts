import { describe, it, expect, beforeEach } from 'vitest';
import { HistoryManager, createHistoryManager } from '../utils/history';

interface TestState {
  value: number;
  items: string[];
}

describe('HistoryManager', () => {
  let history: HistoryManager<TestState>;

  beforeEach(() => {
    // Clear localStorage
    localStorage.clear();
    history = new HistoryManager<TestState>({ maxSize: 5 });
  });

  describe('push', () => {
    it('adds state to history', () => {
      history.push({ value: 1, items: ['a'] });
      expect(history.getCurrentState()).toEqual({ value: 1, items: ['a'] });
    });

    it('clears redo stack on new push', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.undo();
      expect(history.canRedo()).toBe(true);

      history.push({ value: 3, items: [] });
      expect(history.canRedo()).toBe(false);
    });

    it('respects maxSize', () => {
      for (let i = 0; i < 10; i++) {
        history.push({ value: i, items: [] });
      }
      expect(history.getUndoCount()).toBe(4); // maxSize is 5, so 4 undos available
    });

    it('stores action description', () => {
      history.push({ value: 0, items: [] }, 'Initial');
      history.push({ value: 1, items: [] }, 'Set value to 1');
      expect(history.getLastAction()).toBe('Set value to 1');
    });
  });

  describe('undo', () => {
    it('returns null when no history', () => {
      expect(history.undo()).toBeNull();
    });

    it('returns null when only one state', () => {
      history.push({ value: 1, items: [] });
      expect(history.undo()).toBeNull();
    });

    it('returns previous state', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      const result = history.undo();
      expect(result).toEqual({ value: 1, items: [] });
    });

    it('multiple undos work correctly', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.push({ value: 3, items: [] });

      expect(history.undo()).toEqual({ value: 2, items: [] });
      expect(history.undo()).toEqual({ value: 1, items: [] });
      expect(history.undo()).toBeNull();
    });
  });

  describe('redo', () => {
    it('returns null when no redo history', () => {
      expect(history.redo()).toBeNull();
    });

    it('returns next state after undo', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.undo();
      const result = history.redo();
      expect(result).toEqual({ value: 2, items: [] });
    });

    it('multiple redos work correctly', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.push({ value: 3, items: [] });

      history.undo();
      history.undo();

      expect(history.redo()).toEqual({ value: 2, items: [] });
      expect(history.redo()).toEqual({ value: 3, items: [] });
      expect(history.redo()).toBeNull();
    });
  });

  describe('canUndo / canRedo', () => {
    it('canUndo is false with no history', () => {
      expect(history.canUndo()).toBe(false);
    });

    it('canUndo is false with single state', () => {
      history.push({ value: 1, items: [] });
      expect(history.canUndo()).toBe(false);
    });

    it('canUndo is true with multiple states', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      expect(history.canUndo()).toBe(true);
    });

    it('canRedo is false initially', () => {
      expect(history.canRedo()).toBe(false);
    });

    it('canRedo is true after undo', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.undo();
      expect(history.canRedo()).toBe(true);
    });
  });

  describe('clear', () => {
    it('clears all history', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.clear();

      expect(history.canUndo()).toBe(false);
      expect(history.canRedo()).toBe(false);
      expect(history.getCurrentState()).toBeNull();
    });
  });

  describe('transactions', () => {
    it('groups changes into single undo', () => {
      const initialState = { value: 0, items: [] };
      history.push(initialState);

      history.beginTransaction(initialState);
      // These intermediate changes are not pushed
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.push({ value: 3, items: [] });
      history.endTransaction({ value: 10, items: ['final'] }, 'Batch update');

      // Should have only 2 states: initial and final
      expect(history.getUndoCount()).toBe(1);
      expect(history.getCurrentState()).toEqual({ value: 10, items: ['final'] });
      expect(history.undo()).toEqual({ value: 0, items: [] });
    });

    it('cancelTransaction discards changes', () => {
      history.push({ value: 1, items: [] });
      history.beginTransaction({ value: 1, items: [] });
      history.cancelTransaction();

      expect(history.getUndoCount()).toBe(0);
      expect(history.getCurrentState()).toEqual({ value: 1, items: [] });
    });
  });

  describe('persistence', () => {
    it('saves to localStorage when configured', () => {
      const storageHistory = new HistoryManager<TestState>({
        storageKey: 'test-history',
        maxSize: 5,
      });

      storageHistory.push({ value: 1, items: [] });
      storageHistory.push({ value: 2, items: [] });

      const stored = localStorage.getItem('test-history');
      expect(stored).toBeTruthy();
      const parsed = JSON.parse(stored!);
      expect(parsed.past.length).toBe(2);
    });

    it('loads from localStorage on init', () => {
      // First instance saves
      const storageHistory1 = new HistoryManager<TestState>({
        storageKey: 'test-history-2',
        maxSize: 5,
      });
      storageHistory1.push({ value: 1, items: [] });
      storageHistory1.push({ value: 2, items: [] });

      // Second instance loads
      const storageHistory2 = new HistoryManager<TestState>({
        storageKey: 'test-history-2',
        maxSize: 5,
      });
      expect(storageHistory2.getCurrentState()).toEqual({ value: 2, items: [] });
      expect(storageHistory2.canUndo()).toBe(true);
    });

    it('clears localStorage on clear()', () => {
      const storageHistory = new HistoryManager<TestState>({
        storageKey: 'test-history-3',
        maxSize: 5,
      });
      storageHistory.push({ value: 1, items: [] });
      storageHistory.clear();

      expect(localStorage.getItem('test-history-3')).toBeNull();
    });
  });

  describe('createHistoryManager', () => {
    it('creates a history manager instance', () => {
      const hm = createHistoryManager<TestState>({ maxSize: 10 });
      expect(hm).toBeInstanceOf(HistoryManager);
    });
  });

  describe('getUndoCount / getRedoCount', () => {
    it('returns correct counts', () => {
      history.push({ value: 1, items: [] });
      history.push({ value: 2, items: [] });
      history.push({ value: 3, items: [] });

      expect(history.getUndoCount()).toBe(2);
      expect(history.getRedoCount()).toBe(0);

      history.undo();
      expect(history.getUndoCount()).toBe(1);
      expect(history.getRedoCount()).toBe(1);
    });
  });

  describe('deep cloning', () => {
    it('clones state to prevent mutation', () => {
      const original = { value: 1, items: ['a', 'b'] };
      history.push(original);

      original.value = 999;
      original.items.push('c');

      expect(history.getCurrentState()).toEqual({ value: 1, items: ['a', 'b'] });
    });
  });
});
