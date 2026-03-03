import { test, expect } from '@playwright/test';
import { loadPipeline, clickRun } from './helpers';

test.describe('WebSocket Reconnection & Status Persistence', () => {
  test('connection indicator shows green Wifi icon when connected', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // lucide-react renders Wifi as <svg class="lucide lucide-wifi ...">
    // When connected: text-green-500 class
    const footer = page.locator('footer');
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 5_000 });
  });

  test('WebSocket reconnects after page reload', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // Verify initially connected
    const footer = page.locator('footer');
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 5_000 });

    // Reload the page
    await page.reload();
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // After reload, WebSocket should reconnect and show green indicator again
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 10_000 });
  });

  test('navigating away and back re-establishes connection', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    const footer = page.locator('footer');
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 5_000 });

    // Navigate away to a blank page (tears down WebSocket)
    await page.goto('about:blank');

    // Navigate back to the app
    await page.goto('/');
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // Connection should be re-established
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 10_000 });
  });

  test('pipeline modules remain on canvas across tab switches', async ({ page }) => {
    await loadPipeline(page);

    // Verify modules are on the canvas
    const nodes = page.locator('.react-flow__node');
    const initialCount = await nodes.count();
    expect(initialCount).toBeGreaterThan(0);

    // Switch to Logs tab
    await page.getByRole('button', { name: /^Logs/ }).first().click();

    // Switch to Problems tab
    await page.getByRole('button', { name: /^Problems/ }).first().click();

    // Canvas modules should still be present (SPA state preserved)
    const afterCount = await nodes.count();
    expect(afterCount).toBe(initialCount);
  });

  test('status bar shows idle after page reload (client state resets)', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait for running or completed
    const statusSpan = page.locator('footer span.capitalize');
    await expect(async () => {
      const text = await statusSpan.textContent();
      expect(['running', 'completed']).toContain(text);
    }).toPass({ timeout: 15_000 });

    // Reload the page — client Zustand store resets, losing pipelineId
    await page.reload();
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // WebSocket reconnects
    const footer = page.locator('footer');
    await expect(footer.locator('svg.lucide-wifi.text-green-500')).toBeVisible({ timeout: 10_000 });

    // Status resets to idle since client-side state is not persisted across reloads
    await expect(statusSpan).toHaveText('idle', { timeout: 5_000 });
  });
});
