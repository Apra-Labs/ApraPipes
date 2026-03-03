import { test, expect } from '@playwright/test';
import { loadPipeline, clickRun, getLogEntries } from './helpers';

/**
 * Helper: switch to the Logs tab and wait for at least one log row to appear.
 */
async function openLogsAndRun(page: import('@playwright/test').Page) {
  await loadPipeline(page);
  await page.getByRole('button', { name: /^Logs/ }).first().click();
  await clickRun(page);
  // Wait for at least one log row
  await expect(page.locator('.font-mono.text-xs.border-b').first()).toBeVisible({ timeout: 5_000 });
}

test.describe('Log entry content and filtering', () => {
  test('info-level log rows have blue badge styling', async ({ page }) => {
    await openLogsAndRun(page);

    // Find an INF badge — it should have blue styling
    const infBadge = page.locator('.font-mono.text-xs.border-b span.bg-blue-100').first();
    await expect(infBadge).toBeVisible();
    await expect(infBadge).toHaveText('INF');
  });

  test('level filter: clicking Info shows only info entries', async ({ page }) => {
    await openLogsAndRun(page);

    // Click the "Info" level filter button
    const infoButton = page.locator('button', { hasText: /^Info\s*\(/ });
    await infoButton.click();

    // All visible log rows should have INF badges
    const rows = page.locator('.font-mono.text-xs.border-b');
    const count = await rows.count();
    expect(count).toBeGreaterThanOrEqual(1);

    for (let i = 0; i < count; i++) {
      const badge = rows.nth(i).locator('span').nth(1);
      await expect(badge).toHaveText('INF');
    }
  });

  test('level filter: clicking Error shows only error entries or empty', async ({ page }) => {
    await openLogsAndRun(page);

    // Click the "Error" level filter
    const errorButton = page.locator('button', { hasText: /^Error\s*\(/ });
    await errorButton.click();

    const rows = page.locator('.font-mono.text-xs.border-b');
    const count = await rows.count();

    if (count > 0) {
      // If there are error rows, they should all have ERR badges
      for (let i = 0; i < count; i++) {
        const badge = rows.nth(i).locator('span').nth(1);
        await expect(badge).toHaveText('ERR');
      }
    } else {
      // No errors — the "No logs match" placeholder should be visible
      await expect(page.getByText('No logs match the current filters')).toBeVisible();
    }
  });

  test('text search filters log entries', async ({ page }) => {
    await openLogsAndRun(page);

    const searchInput = page.locator('input[placeholder="Search logs..."]');
    await searchInput.fill('pipeline');

    // Wait for filtered results
    await page.waitForTimeout(300);

    const entries = await getLogEntries(page);
    // All visible entries should contain "pipeline" in source or message
    for (const entry of entries) {
      const combined = `${entry.source} ${entry.message}`.toLowerCase();
      expect(combined).toContain('pipeline');
    }
  });

  test('text search shows "N of M" filter count when results are narrowed', async ({ page }) => {
    await openLogsAndRun(page);

    // Get total log count before filtering
    const totalRows = await page.locator('.font-mono.text-xs.border-b').count();

    // Search for "created" — should match only the "Pipeline created" log, not all
    const searchInput = page.locator('input[placeholder="Search logs..."]');
    await searchInput.fill('created');
    await page.waitForTimeout(300);

    const filteredRows = await page.locator('.font-mono.text-xs.border-b').count();

    if (filteredRows < totalRows) {
      // "N of M" text should be visible when filtering reduces the count
      const filterCount = page.locator('span.text-gray-500', { hasText: /\d+ of \d+/ });
      await expect(filterCount).toBeVisible();
      const text = await filterCount.textContent();
      expect(text).toMatch(new RegExp(`${filteredRows} of ${totalRows}`));
    }
    // If all logs happen to match, filter count won't appear — that's valid
    expect(filteredRows).toBeGreaterThanOrEqual(1);
  });

  test('clear button removes all logs', async ({ page }) => {
    await openLogsAndRun(page);

    // Confirm logs exist
    const rows = page.locator('.font-mono.text-xs.border-b');
    await expect(rows.first()).toBeVisible();

    // Click Clear button
    const clearButton = page.locator('button', { hasText: 'Clear' });
    await clearButton.click();

    // Logs should be empty — placeholder text appears
    await expect(page.getByText('No logs yet')).toBeVisible({ timeout: 2_000 });
    expect(await rows.count()).toBe(0);
  });

  test('clicking All filter resets after filtering by level', async ({ page }) => {
    await openLogsAndRun(page);

    // Get total count before filtering
    const rowsBefore = await page.locator('.font-mono.text-xs.border-b').count();

    // Filter by Info
    const infoButton = page.locator('button', { hasText: /^Info\s*\(/ });
    await infoButton.click();

    const rowsFiltered = await page.locator('.font-mono.text-xs.border-b').count();

    // Click All to reset
    const allButton = page.locator('button', { hasText: /^All\s*\(/ });
    await allButton.click();

    const rowsAfter = await page.locator('.font-mono.text-xs.border-b').count();
    expect(rowsAfter).toBe(rowsBefore);
  });
});
