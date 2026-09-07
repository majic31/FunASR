import { expect, test } from '@playwright/test';

for (const width of [390, 1440]) {
  for (const language of ['zh', 'en']) {
    test(`Model selection to raw tag recipe: ${language} ${width}px`, async ({ page }, testInfo) => {
      const prefix = language === 'en' ? '/en' : '';
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.setViewportSize({ width, height: 900 });
      await page.goto(`${prefix}/docs/model-selection.html`);
      const recipe = page.locator(`.docs-article a[href="${prefix}/docs/speaker-emotion.html"]`);
      await expect(recipe).toHaveCount(1);
      await recipe.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      await page.screenshot({ path: testInfo.outputPath('http-aliases.png') });
      await recipe.click();
      await expect(page).toHaveURL(new RegExp(`${prefix}/docs/speaker-emotion.html$`));
      const code = page.locator('.docs-article pre').filter({ hasText: 'raw_tagged_text' });
      await expect(code).toHaveCount(1);
      await code.scrollIntoViewIfNeeded();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
      expect(errors).toEqual([]);
    });
  }
}
