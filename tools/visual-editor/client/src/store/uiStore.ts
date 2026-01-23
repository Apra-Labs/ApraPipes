import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ViewMode = 'visual' | 'json' | 'split';

interface Settings {
  validateOnSave: boolean;
}

interface UIState {
  viewMode: ViewMode;
  leftPanelCollapsed: boolean;
  rightPanelCollapsed: boolean;
  settingsDialogVisible: boolean;
  settings: Settings;
}

interface UIActions {
  setViewMode: (mode: ViewMode) => void;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
  openSettingsDialog: () => void;
  closeSettingsDialog: () => void;
  setValidateOnSave: (value: boolean) => void;
}

const initialState: UIState = {
  viewMode: 'visual',
  leftPanelCollapsed: false,
  rightPanelCollapsed: false,
  settingsDialogVisible: false,
  settings: {
    validateOnSave: true,
  },
};

/**
 * UI store for managing application view state
 */
export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set) => ({
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

      openSettingsDialog: () => {
        set({ settingsDialogVisible: true });
      },

      closeSettingsDialog: () => {
        set({ settingsDialogVisible: false });
      },

      setValidateOnSave: (value) => {
        set((state) => ({
          settings: { ...state.settings, validateOnSave: value },
        }));
      },
    }),
    {
      name: 'aprapipes-studio-ui',
      partialize: (state) => ({ settings: state.settings }),
    }
  )
);
