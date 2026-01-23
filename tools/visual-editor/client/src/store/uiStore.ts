import { create } from 'zustand';

export type ViewMode = 'visual' | 'json' | 'split';

interface UIState {
  viewMode: ViewMode;
  leftPanelCollapsed: boolean;
  rightPanelCollapsed: boolean;
}

interface UIActions {
  setViewMode: (mode: ViewMode) => void;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
}

const initialState: UIState = {
  viewMode: 'visual',
  leftPanelCollapsed: false,
  rightPanelCollapsed: false,
};

/**
 * UI store for managing application view state
 */
export const useUIStore = create<UIState & UIActions>((set) => ({
  ...initialState,

  setViewMode: (mode) => {
    set({ viewMode: mode });
  },

  toggleLeftPanel: () => {
    set((state) => ({ leftPanelCollapsed: !state.leftPanelCollapsed }));
  },

  toggleRightPanel: () => {
    set((state) => ({ rightPanelCollapsed: !state.rightPanelCollapsed }));
  },
}));
