/**
 * E2E Test Helpers
 *
 * Reusable helpers for Playwright e2e tests of the ApraPipes Visual Editor.
 */

import { type Page, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const DEFAULT_PIPELINE = path.resolve(__dirname, 'fixtures/pipeline.json');

/**
 * Load a pipeline into the visual editor via the Import dialog.
 * Opens the app, clicks Import, pastes the JSON, confirms, and waits
 * for module nodes to appear on the canvas.
 */
export async function loadPipeline(
  page: Page,
  pipelinePath: string = process.env.E2E_PIPELINE || DEFAULT_PIPELINE,
): Promise<void> {
  const json = fs.readFileSync(pipelinePath, 'utf-8');

  await page.goto('/');
  await expect(page.getByText('ApraPipes Studio')).toBeVisible();

  // Open the Import dialog
  await page.getByRole('button', { name: 'Import' }).click();
  await expect(page.getByText('Import Pipeline JSON')).toBeVisible();

  // Paste the pipeline JSON and confirm
  await page.locator('textarea[placeholder="Paste pipeline JSON here..."]').fill(json);
  // Click the dialog's Import button (inside the fixed overlay, not the toolbar)
  await page.locator('div.fixed button').filter({ hasText: 'Import' }).click();

  // Wait for at least one module node to appear on the canvas
  await expect(page.locator('.react-flow__node').first()).toBeVisible({ timeout: 5000 });
}

/**
 * Wait for the status bar to show the expected status text.
 *
 * The StatusBar renders status in a capitalized span inside a <footer> element,
 * e.g. "idle", "running", "stopped", "completed", "error", "creating", "stopping".
 */
export async function waitForStatus(
  page: Page,
  status: string,
  timeout: number = 10_000,
): Promise<void> {
  const statusSpan = page.locator('footer span.capitalize');
  await expect(statusSpan).toHaveText(status, { timeout });
}

/**
 * Get all visible log entries from the Logs panel.
 * Clicks the Logs tab first to ensure it's active.
 *
 * Each LogRow renders 4 child spans: timestamp, level badge, source, message.
 */
export async function getLogEntries(
  page: Page,
): Promise<{ timestamp: string; level: string; source: string; message: string }[]> {
  // Switch to Logs tab (click the first matching button since the tab label contains "Logs")
  await page.getByRole('button', { name: /^Logs/ }).first().click();

  // Log rows are direct children of the scrollable container inside LogsPanel.
  // Each row: div.flex.items-start with 4 span children.
  const rows = page.locator('.font-mono.text-xs.border-b');
  const count = await rows.count();

  const entries: { timestamp: string; level: string; source: string; message: string }[] = [];
  for (let i = 0; i < count; i++) {
    const row = rows.nth(i);
    const spans = row.locator('> span');
    const spanCount = await spans.count();
    if (spanCount >= 4) {
      entries.push({
        timestamp: (await spans.nth(0).textContent()) || '',
        level: (await spans.nth(1).textContent())?.trim() || '',
        source: (await spans.nth(2).textContent())?.trim() || '',
        message: (await spans.nth(3).textContent())?.trim() || '',
      });
    }
  }
  return entries;
}

/**
 * Get the log count displayed on the Logs tab badge.
 * Returns 0 if no badge is visible (meaning zero logs).
 */
export async function getLogCount(page: Page): Promise<number> {
  const badge = page.getByRole('button', { name: /^Logs/ }).first().locator('span.rounded-full');
  if (await badge.isVisible()) {
    const text = await badge.textContent();
    return parseInt(text || '0', 10);
  }
  return 0;
}

/**
 * Get all visible items from the Problems panel.
 */
export async function getProblems(page: Page): Promise<string[]> {
  // Switch to Problems tab
  await page.getByRole('button', { name: /^Problems/ }).first().click();

  const items = page.locator('button[class*="border-l-"]');
  const count = await items.count();
  const problems: string[] = [];
  for (let i = 0; i < count; i++) {
    const text = await items.nth(i).textContent();
    if (text) problems.push(text.trim());
  }
  return problems;
}

/**
 * Click the Run button in the toolbar.
 */
export async function clickRun(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Run' }).click();
}

/**
 * Click the Stop button in the toolbar.
 */
export async function clickStop(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Stop' }).click();
}

/**
 * Get the number of module nodes visible on the canvas.
 */
export async function getModuleCount(page: Page): Promise<number> {
  return page.locator('.react-flow__node').count();
}
