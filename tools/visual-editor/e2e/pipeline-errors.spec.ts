import { test, expect } from '@playwright/test';
import { loadPipeline } from './helpers';
import * as fs from 'fs';
import * as path from 'path';

const BAD_PIPELINE = path.resolve(__dirname, 'fixtures/bad-pipeline.json');

/**
 * Helper: open the app, import bad-pipeline.json via the Import dialog.
 * Unlike loadPipeline, this doesn't require all modules to render on the canvas.
 */
async function importBadPipeline(page: import('@playwright/test').Page) {
  const json = fs.readFileSync(BAD_PIPELINE, 'utf-8');

  await page.goto('/');
  await expect(page.getByText('ApraPipes Studio')).toBeVisible();

  // Open the Import dialog
  await page.getByRole('button', { name: 'Import' }).click();
  await expect(page.getByText('Import Pipeline JSON')).toBeVisible();

  // Paste the bad pipeline JSON and confirm
  await page.locator('textarea[placeholder="Paste pipeline JSON here..."]').fill(json);
  await page.locator('div.fixed button').filter({ hasText: 'Import' }).click();

  // Wait for at least the valid module to render (TestSignalGenerator)
  await expect(page.locator('.react-flow__node').first()).toBeVisible({ timeout: 5000 });
}

/**
 * Helper: click Validate inside the Problems panel (uses the panel-specific button,
 * not the toolbar one) and wait for validation to complete.
 */
async function clickValidateInProblemsPanel(page: import('@playwright/test').Page) {
  // The Problems panel validate button is the small blue one with specific styling
  const validateBtn = page.locator('button.bg-blue-500', { hasText: 'Validate' });
  await validateBtn.click();
  await expect(validateBtn).toBeEnabled({ timeout: 10_000 });
}

test.describe('Error Scenarios', () => {
  test('malformed JSON import shows error alert', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('ApraPipes Studio')).toBeVisible();

    // Open the Import dialog
    await page.getByRole('button', { name: 'Import' }).click();
    await expect(page.getByText('Import Pipeline JSON')).toBeVisible();

    // Paste malformed JSON
    await page.locator('textarea[placeholder="Paste pipeline JSON here..."]').fill('{ this is not valid json !!!');

    // Set up dialog handler BEFORE triggering it (alert blocks the click)
    page.on('dialog', (dialog) => dialog.accept());

    // Click Import — this triggers an alert() for malformed JSON
    await page.locator('div.fixed button').filter({ hasText: 'Import' }).click();

    // The import dialog should still be visible (import failed, dialog stays open)
    await expect(page.getByText('Import Pipeline JSON')).toBeVisible();
  });

  test('pipeline with unknown module type shows only valid modules on canvas', async ({ page }) => {
    await importBadPipeline(page);

    // bad-pipeline.json has 2 modules: NonExistentModuleType (skipped) + TestSignalGenerator (rendered)
    // Only the valid module should appear on the canvas
    const nodeCount = await page.locator('.react-flow__node').count();
    expect(nodeCount).toBe(1);

    // The valid module (TestSignalGenerator) should be visible
    await expect(page.getByText('TestSignalGenerator').first()).toBeVisible();
  });

  test('validation detects unknown module type error', async ({ page }) => {
    await importBadPipeline(page);

    // Switch to Problems tab
    await page.getByRole('button', { name: /^Problems/ }).first().click();

    // Validate using the Problems panel button
    await clickValidateInProblemsPanel(page);

    // Should have at least one error for the unknown module type (E100)
    const errorIssues = page.locator('button.w-full.text-left span.font-mono');
    await expect(errorIssues.first()).toBeVisible({ timeout: 5_000 });

    // Look for E100 error code (Unknown module type)
    await expect(page.getByText('E100')).toBeVisible();
  });

  test('validation detects broken connection references', async ({ page }) => {
    await importBadPipeline(page);

    // Switch to Problems tab and validate
    await page.getByRole('button', { name: /^Problems/ }).first().click();
    await clickValidateInProblemsPanel(page);

    // bad-pipeline.json has a connection to "missing_target" which doesn't exist
    // Should produce E301 (unknown destination module)
    await expect(page.getByText('E301')).toBeVisible();
  });

  test('validation errors show proper formatting (code, message, location)', async ({ page }) => {
    await importBadPipeline(page);

    // Switch to Problems tab and validate
    await page.getByRole('button', { name: /^Problems/ }).first().click();
    await clickValidateInProblemsPanel(page);

    // Each issue row should have: error code (font-mono), message text, and location
    const issueRows = page.locator('button.w-full.text-left');
    const rowCount = await issueRows.count();
    expect(rowCount).toBeGreaterThanOrEqual(1);

    // Check the first issue has the expected structure
    const firstRow = issueRows.first();
    // Error code in font-mono span
    const code = firstRow.locator('span.font-mono').first();
    await expect(code).toBeVisible();
    const codeText = await code.textContent();
    expect(codeText).toMatch(/^[EWI]\d{3}$/);

    // Message text
    const message = firstRow.locator('.text-sm.text-gray-900');
    await expect(message).toBeVisible();
    const messageText = await message.textContent();
    expect(messageText!.length).toBeGreaterThan(0);

    // Location in font-mono
    const location = firstRow.locator('.text-xs.text-gray-500.font-mono');
    await expect(location).toBeVisible();
  });

  test('Errors filter in Problems tab shows only errors', async ({ page }) => {
    await importBadPipeline(page);

    // Switch to Problems tab and validate
    await page.getByRole('button', { name: /^Problems/ }).first().click();
    await clickValidateInProblemsPanel(page);

    // Click Errors filter button
    const errorsFilter = page.locator('button', { hasText: /^Errors\s*\(/ });
    await expect(errorsFilter).toBeVisible();
    await errorsFilter.click();

    // All visible issues should be errors (have error codes starting with E)
    const issueCodes = page.locator('button.w-full.text-left span.font-mono');
    const count = await issueCodes.count();
    expect(count).toBeGreaterThanOrEqual(1);

    for (let i = 0; i < count; i++) {
      const text = await issueCodes.nth(i).textContent();
      expect(text).toMatch(/^E\d{3}$/);
    }
  });

  test('valid pipeline shows no errors after validation', async ({ page }) => {
    // Load the good pipeline
    await loadPipeline(page);

    // Switch to Problems tab and validate
    await page.getByRole('button', { name: /^Problems/ }).first().click();
    await clickValidateInProblemsPanel(page);

    // Should show success message (no issues)
    await expect(page.getByText('No issues found')).toBeVisible({ timeout: 5_000 });
  });
});
