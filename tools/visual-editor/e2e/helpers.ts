import { Page, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const DEFAULT_PIPELINE = path.resolve(__dirname, 'fixtures/pipeline.json');

/**
 * Load a pipeline into the visual editor via the Import dialog.
 * Opens the app, clicks Import, pastes the JSON, and confirms.
 */
export async function loadPipeline(
  page: Page,
  pipelinePath: string = process.env.E2E_PIPELINE || DEFAULT_PIPELINE
): Promise<void> {
  const json = fs.readFileSync(pipelinePath, 'utf-8');

  await page.goto('/');
  await expect(page.getByText('ApraPipes Studio')).toBeVisible();

  // Open the Import dialog
  await page.getByRole('button', { name: 'Import' }).click();
  await expect(page.getByText('Import Pipeline JSON')).toBeVisible();

  // Paste the pipeline JSON and confirm
  await page.locator('textarea[placeholder="Paste pipeline JSON here..."]').fill(json);
  // Click the dialog's Import button (inside the dialog, not the toolbar)
  await page.locator('div.fixed button').filter({ hasText: 'Import' }).click();
}

/**
 * Wait for the status bar to show the expected status text.
 */
export async function waitForStatus(
  page: Page,
  status: string,
  timeout: number = 10_000
): Promise<void> {
  await expect(
    page.locator('footer').getByText(status, { exact: false })
  ).toBeVisible({ timeout });
}

/**
 * Get all visible log entries from the Logs panel.
 * Clicks the Logs tab first if it isn't already active.
 */
export async function getLogEntries(
  page: Page
): Promise<{ timestamp: string; level: string; source: string; message: string }[]> {
  // Switch to Logs tab
  await page.getByRole('button', { name: /Logs/ }).click();

  // Each log row is a flex div with timestamp, level badge, source, and message spans
  const rows = page.locator('[class*="font-mono text-xs"]').filter({ has: page.locator('span') });
  const count = await rows.count();

  const entries: { timestamp: string; level: string; source: string; message: string }[] = [];
  for (let i = 0; i < count; i++) {
    const row = rows.nth(i);
    const text = await row.textContent();
    if (text) {
      // Rows are rendered as: "HH:MM:SS.mmm  LVL  source  message"
      entries.push({
        timestamp: (await row.locator('span').first().textContent()) || '',
        level: (await row.locator('span').nth(1).textContent()) || '',
        source: (await row.locator('span').nth(2).textContent()) || '',
        message: (await row.locator('span').nth(3).textContent()) || '',
      });
    }
  }
  return entries;
}

/**
 * Get all visible items from the Problems panel.
 */
export async function getProblems(
  page: Page
): Promise<string[]> {
  // Switch to Problems tab
  await page.getByRole('button', { name: /Problems/ }).click();

  const items = page.locator('button[class*="border-l-"]');
  const count = await items.count();
  const problems: string[] = [];
  for (let i = 0; i < count; i++) {
    const text = await items.nth(i).textContent();
    if (text) problems.push(text.trim());
  }
  return problems;
}
