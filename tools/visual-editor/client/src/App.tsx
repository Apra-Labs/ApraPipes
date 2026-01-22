import { useEffect, useState } from 'react';
import { Toolbar } from './components/Toolbar/Toolbar';
import { StatusBar } from './components/Toolbar/StatusBar';
import { ModulePalette } from './components/Panels/ModulePalette';
import { api } from './services/api';
import type { ModuleSchema } from './types/schema';

/**
 * Main application component for ApraPipes Studio
 * Layout: Toolbar (top) | Main Area (center) | Status Bar (bottom)
 */
function App() {
  const [schema, setSchema] = useState<Record<string, ModuleSchema> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSchema = async () => {
      try {
        setLoading(true);
        const data = await api.getSchema();
        setSchema(data.modules);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load schema');
      } finally {
        setLoading(false);
      }
    };

    loadSchema();
  }, []);

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

        {/* Center: Canvas (placeholder) */}
        <main className="flex-1 bg-gray-50 flex items-center justify-center">
          <div className="text-muted-foreground text-center">
            <p className="text-lg font-medium">Canvas Area</p>
            <p className="text-sm">Drag modules here to build your pipeline</p>
          </div>
        </main>

        {/* Right Panel: Property Panel (placeholder) */}
        <aside className="w-72 border-l border-border bg-muted/30 p-4">
          <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide mb-4">
            Properties
          </h3>
          <p className="text-sm text-muted-foreground">
            Select a module to view its properties
          </p>
        </aside>
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}

export default App;
