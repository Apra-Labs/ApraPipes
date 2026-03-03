import { test, expect } from '@playwright/test';

test('app loads and shows heading', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/ApraPipes Studio/);
  await expect(page.getByText('ApraPipes Studio')).toBeVisible();
});
