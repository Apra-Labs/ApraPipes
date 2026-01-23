import { useMemo } from 'react';
import Editor from '@monaco-editor/react';
import { usePipelineStore } from '../../store/pipelineStore';

interface JsonViewProps {
  readOnly?: boolean;
}

/**
 * JSON View component displaying pipeline configuration with Monaco Editor
 */
export function JsonView({ readOnly = true }: JsonViewProps) {
  const config = usePipelineStore((state) => state.config);
  const fromJSON = usePipelineStore((state) => state.fromJSON);

  // Format the JSON for display
  const jsonValue = useMemo(() => {
    return JSON.stringify(config, null, 2);
  }, [config]);

  const handleEditorChange = (value: string | undefined) => {
    if (!readOnly && value) {
      try {
        fromJSON(value);
      } catch {
        // Silently ignore parse errors while typing
      }
    }
  };

  return (
    <div className="h-full w-full" data-testid="json-view">
      <Editor
        height="100%"
        language="json"
        theme="vs-light"
        value={jsonValue}
        onChange={handleEditorChange}
        options={{
          readOnly,
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          formatOnPaste: true,
          formatOnType: true,
          automaticLayout: true,
          tabSize: 2,
        }}
      />
    </div>
  );
}
