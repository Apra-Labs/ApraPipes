import { test, expect } from '@playwright/test';
import { loadPipeline, waitForStatus, getModuleCount, getProblems } from './helpers';

test.describe('Pipeline Load', () => {
  test('loads pipeline and shows modules on canvas', async ({ page }) => {
    await loadPipeline(page);

    // Verify modules appear on the canvas
    const moduleCount = await getModuleCount(page);
    expect(moduleCount).toBe(4); // generator, color_convert, bmp_encoder, writer
  });

  test('shows IDLE status after loading', async ({ page }) => {
    await loadPipeline(page);

    // Status bar should show "idle" (lowercase, capitalized via CSS)
    await waitForStatus(page, 'idle');
  });

  test('Problems tab is accessible', async ({ page }) => {
    await loadPipeline(page);

    // Click Problems tab — it should be visible and clickable
    const problemsTab = page.getByRole('button', { name: /^Problems/ }).first();
    await expect(problemsTab).toBeVisible();
    await problemsTab.click();

    // The problems panel should be visible (even if empty)
    // An empty problems panel shows placeholder text or an empty list
    const panel = page.locator('.border-t.border-gray-200');
    await expect(panel).toBeVisible();
  });

  test('canvas shows correct module types', async ({ page }) => {
    await loadPipeline(page);

    // Verify each expected module appears
    const nodes = page.locator('.react-flow__node');
    const count = await nodes.count();
    expect(count).toBe(4);

    // Check that module names/types are visible on the nodes
    await expect(page.getByText('TestSignalGenerator').first()).toBeVisible();
    await expect(page.getByText('ColorConversion').first()).toBeVisible();
    await expect(page.getByText('BMPConverter').first()).toBeVisible();
    await expect(page.getByText('FileWriterModule').first()).toBeVisible();
  });

  test('connections are rendered between modules', async ({ page }) => {
    await loadPipeline(page);

    // ReactFlow renders edges as SVG elements
    const edges = page.locator('.react-flow__edge');
    const edgeCount = await edges.count();
    expect(edgeCount).toBe(3); // 3 connections in the fixture
  });
});
