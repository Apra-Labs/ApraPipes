import { test, expect } from '@playwright/test';
import { loadPipeline, waitForStatus, clickRun, clickStop } from './helpers';

test.describe('Pipeline Run & Status Transitions', () => {
  test('clicking Run transitions status from idle', async ({ page }) => {
    await loadPipeline(page);
    await waitForStatus(page, 'idle');

    // Click Run — this creates the pipeline on the server and starts it
    await clickRun(page);

    // Status should transition away from "idle" — could be "creating", "running", or "error"
    // Wait for any non-idle status
    const statusSpan = page.locator('footer span.capitalize');
    await expect(statusSpan).not.toHaveText('idle', { timeout: 10_000 });

    // Capture whatever status we reached
    const status = await statusSpan.textContent();
    expect(['creating', 'running', 'completed', 'error']).toContain(status);
  });

  test('pipeline reaches running state', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait for RUNNING or COMPLETED (pipeline may complete very fast with 10 frames)
    const statusSpan = page.locator('footer span.capitalize');
    await expect(async () => {
      const text = await statusSpan.textContent();
      expect(['running', 'completed']).toContain(text);
    }).toPass({ timeout: 15_000 });
  });

  test('Run button disables while pipeline is active', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait for non-idle status
    const statusSpan = page.locator('footer span.capitalize');
    await expect(statusSpan).not.toHaveText('idle', { timeout: 10_000 });

    // Run button should be disabled while running (title changes to "Running...")
    const runButton = page.locator('header button', { hasText: /^Run/ }).first();
    await expect(runButton).toBeDisabled();
  });

  test('Stop button stops a running pipeline', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait for RUNNING
    const statusSpan = page.locator('footer span.capitalize');
    await expect(async () => {
      const text = await statusSpan.textContent();
      expect(['running', 'completed']).toContain(text);
    }).toPass({ timeout: 15_000 });

    const currentStatus = await statusSpan.textContent();

    if (currentStatus === 'running') {
      // Stop the pipeline
      await clickStop(page);

      // After stop + delete, status should return to idle
      await waitForStatus(page, 'idle', 10_000);
    } else {
      // Pipeline completed on its own (maxFrames=10 is fast)
      expect(currentStatus).toBe('completed');
    }
  });

  test('status bar shows duration while running', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait for running
    const statusSpan = page.locator('footer span.capitalize');
    await expect(async () => {
      const text = await statusSpan.textContent();
      expect(['running', 'completed']).toContain(text);
    }).toPass({ timeout: 15_000 });

    // Duration timer should be visible (format: MM:SS or HH:MM:SS)
    const footer = page.locator('footer');
    await expect(footer.getByText(/\d{2}:\d{2}/)).toBeVisible({ timeout: 5_000 });
  });
});
