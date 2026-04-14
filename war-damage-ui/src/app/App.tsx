import { Component, type ReactNode } from 'react';
import { RouterProvider } from 'react-router';
import { router } from './routes';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      const err = this.state.error as Error;
      return (
        <div style={{ padding: 32, fontFamily: 'monospace', color: '#e24b4a', backgroundColor: '#0d0f14', minHeight: '100vh' }}>
          <div style={{ marginBottom: 8, fontSize: 12, color: '#888780', letterSpacing: '0.1em' }}>RENDER ERROR</div>
          <div style={{ marginBottom: 16 }}>{err.message}</div>
          <pre style={{ fontSize: 11, color: '#888780', whiteSpace: 'pre-wrap' }}>{err.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}
