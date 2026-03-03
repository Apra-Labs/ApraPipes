import { test, expect } from '@playwright/test';
import { loadPipeline, clickRun, clickStop, getLogEntries, getLogCount } from './helpers';

test.describe('Logs Tab — entries after pipeline run', () => {
  test('logs appear after running a pipeline', async ({ page }) => {
    await loadPipeline(page);

    // Switch to Logs tab before running
    await page.getByRole('button', { name: /^Logs/ }).first().click();

    await clickRun(page);

    // Wait for at least one log entry to appear (up to 5 seconds)
    const logRows = page.locator('.font-mono.text-xs.border-b');
    await expect(logRows.first()).toBeVisible({ timeout: 5_000 });

    const entries = await getLogEntries(page);
    expect(entries.length).toBeGreaterThanOrEqual(1);
  });

  test('log entries have timestamp, level, source, and message', async ({ page }) => {
    await loadPipeline(page);
    await page.getByRole('button', { name: /^Logs/ }).first().click();
    await clickRun(page);

    // Wait for log rows
    const logRows = page.locator('.font-mono.text-xs.border-b');
    await expect(logRows.first()).toBeVisible({ timeout: 5_000 });

    const entries = await getLogEntries(page);
    expect(entries.length).toBeGreaterThanOrEqual(1);

    for (const entry of entries) {
      // Timestamp: HH:MM:SS.mmm format
      expect(entry.timestamp).toMatch(/\d{2}:\d{2}:\d{2}\.\d{3}/);
      // Level badge: DBG, INF, WRN, or ERR
      expect(entry.level).toMatch(/^(DBG|INF|WRN|ERR)$/);
      // Source should be non-empty
      expect(entry.source.length).toBeGreaterThan(0);
      // Message should be non-empty
      expect(entry.message.length).toBeGreaterThan(0);
    }
  });

  test('"Pipeline created" message appears in logs', async ({ page }) => {
    await loadPipeline(page);
    await page.getByRole('button', { name: /^Logs/ }).first().click();
    await clickRun(page);

    // Wait for the "Pipeline created" log message
    const logRows = page.locator('.font-mono.text-xs.border-b');
    await expect(logRows.first()).toBeVisible({ timeout: 5_000 });

    const entries = await getLogEntries(page);
    const createdMsg = entries.find(e =>
      e.message.toLowerCase().includes('pipeline created'),
    );
    expect(createdMsg).toBeDefined();
    expect(createdMsg!.level).toBe('INF');
    expect(createdMsg!.source).toBe('pipeline');
  });

  test('"Starting pipeline" message appears in logs', async ({ page }) => {
    await loadPipeline(page);
    await page.getByRole('button', { name: /^Logs/ }).first().click();
    await clickRun(page);

    // Wait for logs to populate
    const logRows = page.locator('.font-mono.text-xs.border-b');
    await expect(logRows.first()).toBeVisible({ timeout: 5_000 });

    const entries = await getLogEntries(page);
    const startMsg = entries.find(e =>
      e.message.toLowerCase().includes('starting pipeline'),
    );
    expect(startMsg).toBeDefined();
    expect(startMsg!.level).toBe('INF');
  });

  test('log count badge updates on Logs tab', async ({ page }) => {
    await loadPipeline(page);
    await clickRun(page);

    // Wait a moment for logs to arrive
    await page.waitForTimeout(2_000);

    const count = await getLogCount(page);
    expect(count).toBeGreaterThanOrEqual(1);
  });

  test('browser console has no errors during pipeline run', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await loadPipeline(page);
    await clickRun(page);

    // Wait for pipeline to reach running or completed
    const statusSpan = page.locator('footer span.capitalize');
    await expect(async () => {
      const text = await statusSpan.textContent();
      expect(['running', 'completed']).toContain(text);
    }).toPass({ timeout: 15_000 });

    // Filter out known benign errors (e.g., favicon, HMR)
    const realErrors = consoleErrors.filter(
      e => !e.includes('favicon') && !e.includes('HMR') && !e.includes('404'),
    );
    expect(realErrors).toEqual([]);
  });
});
