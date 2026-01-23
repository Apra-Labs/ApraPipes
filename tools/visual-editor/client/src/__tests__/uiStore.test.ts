import { describe, it, expect, beforeEach } from 'vitest';
import { useUIStore } from '../store/uiStore';

describe('uiStore', () => {
  beforeEach(() => {
    // Reset to initial state
    useUIStore.setState({
      viewMode: 'visual',
      leftPanelCollapsed: false,
      rightPanelCollapsed: false,
    });
  });

  const getState = () => useUIStore.getState();

  describe('setViewMode', () => {
    it('changes view mode to json', () => {
      getState().setViewMode('json');
      expect(getState().viewMode).toBe('json');
    });

    it('changes view mode to split', () => {
      getState().setViewMode('split');
      expect(getState().viewMode).toBe('split');
    });

    it('changes view mode back to visual', () => {
      getState().setViewMode('json');
      getState().setViewMode('visual');
      expect(getState().viewMode).toBe('visual');
    });
  });

  describe('toggleLeftPanel', () => {
    it('collapses left panel', () => {
      expect(getState().leftPanelCollapsed).toBe(false);
      getState().toggleLeftPanel();
      expect(getState().leftPanelCollapsed).toBe(true);
    });

    it('expands left panel', () => {
      getState().toggleLeftPanel();
      getState().toggleLeftPanel();
      expect(getState().leftPanelCollapsed).toBe(false);
    });
  });

  describe('toggleRightPanel', () => {
    it('collapses right panel', () => {
      expect(getState().rightPanelCollapsed).toBe(false);
      getState().toggleRightPanel();
      expect(getState().rightPanelCollapsed).toBe(true);
    });

    it('expands right panel', () => {
      getState().toggleRightPanel();
      getState().toggleRightPanel();
      expect(getState().rightPanelCollapsed).toBe(false);
    });
  });
});
