import { useEffect, useState } from 'react';
import { Toolbar } from './components/Toolbar/Toolbar';
import { StatusBar } from './components/Toolbar/StatusBar';
import { ModulePalette } from './components/Panels/ModulePalette';
import { PropertyPanel } from './components/Panels/PropertyPanel';
import { JsonView } from './components/Panels/JsonView';
import { Canvas } from './components/Canvas/Canvas';
import { api } from './services/api';
import { usePipelineStore } from './store/pipelineStore';
import { useUIStore } from './store/uiStore';
import type { ModuleSchema } from './types/schema';

/**
 * Main application component for ApraPipes Studio
 * Layout: Toolbar (top) | Main Area (center) | Status Bar (bottom)
 */
function App() {
  const [schema, setSchema] = useState<Record<string, ModuleSchema> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const setSchemaInStore = usePipelineStore((state) => state.setSchema);
  const viewMode = useUIStore((state) => state.viewMode);

  useEffect(() => {
    const loadSchema = async () => {
      try {
        setLoading(true);
        const data = await api.getSchema();
        setSchema(data.modules);
        setSchemaInStore(data.modules);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load schema');
      } finally {
        setLoading(false);
      }
    };

    loadSchema();
  }, [setSchemaInStore]);

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Toolbar */}
      <Toolbar />

      {/* Main Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Module Palette */}
        <aside className="w-64 border-r border-border bg-muted/30 overflow-y-auto">
          {loading && (
            <div className="p-4 text-center text-muted-foreground">
              Loading modules...
            </div>
          )}
          {error && (
            <div className="p-4 text-center text-destructive">
              {error}
            </div>
          )}
          {schema && <ModulePalette modules={schema} />}
        </aside>

        {/* Center: Canvas and/or JSON View based on viewMode */}
        {viewMode === 'visual' && (
          <main className="flex-1 bg-gray-50">
            <Canvas schema={schema} />
          </main>
        )}

        {viewMode === 'json' && (
          <main className="flex-1 bg-gray-50">
            <JsonView />
          </main>
        )}

        {viewMode === 'split' && (
          <>
            <main className="flex-1 bg-gray-50">
              <Canvas schema={schema} />
            </main>
            <div className="w-px bg-border" />
            <main className="w-1/3 bg-gray-50">
              <JsonView />
            </main>
          </>
        )}

        {/* Right Panel: Property Panel (visible in visual and split modes) */}
        {viewMode !== 'json' && (
          <aside className="w-72 border-l border-border bg-muted/30 overflow-y-auto">
            <PropertyPanel schema={schema} />
          </aside>
        )}
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}

export default App;
