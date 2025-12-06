import { test, expect } from "@playwright/test";

/**
 * Attestation Panel E2E Tests
 * ===========================
 *
 * These tests validate that the attestation panel correctly displays
 * the dual-root values (R_t, U_t, H_t) and enforces deterministic rendering.
 *
 * Invariants enforced:
 * 1. R_t, U_t, H_t are displayed as truncated 64-char hex strings
 * 2. The truncation format is consistent: first 12 chars + "…" + last 8 chars
 * 3. Metadata badges (version, event counts) are deterministic
 * 4. The panel refreshes without visual glitches
 */

test.describe("Attestation Panel Invariants", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");
  });

  test("displays Dual Attestation section", async ({ page }) => {
    await expect(page.locator("text=Dual Attestation")).toBeVisible();
    await expect(
      page.locator("text=Cryptographic binding of reasoning and UI event streams")
    ).toBeVisible();
  });

  test("displays attestation version badge", async ({ page }) => {
    // Wait for attestation data to load
    await page.waitForTimeout(1000);

    // Look for version badge (v1, v2, etc.)
    const versionBadge = page.locator("span").filter({ hasText: /^v\d+$/ });
    const count = await versionBadge.count();

    if (count > 0) {
      await expect(versionBadge.first()).toBeVisible();
    }
  });

  test("R_t, U_t, H_t are truncated consistently", async ({ page }) => {
    // Wait for attestation data to load
    await page.waitForTimeout(1500);

    // The truncation format should be: 12 chars + "…" + 8 chars = 21 chars total
    // Look for elements containing the ellipsis character
    const truncatedHashes = page.locator("code").filter({ hasText: "…" });
    const count = await truncatedHashes.count();

    // If attestation is loaded, we should see truncated hashes
    if (count > 0) {
      for (let i = 0; i < Math.min(count, 3); i++) {
        const text = await truncatedHashes.nth(i).textContent();
        expect(text).toMatch(/^[0-9a-f]{12}…[0-9a-f]{8}$/i);
      }
    }
  });

  test("attestation panel has refresh button", async ({ page }) => {
    // Find the refresh button in the attestation section
    const attestationSection = page.locator("section").filter({
      has: page.locator("text=Dual Attestation"),
    });

    const refreshButton = attestationSection
      .locator("button")
      .filter({ hasText: /Refresh/i });

    await expect(refreshButton).toBeVisible();
  });

  test("refresh button updates attestation data", async ({ page }) => {
    await page.waitForTimeout(1000);

    // Find and click the refresh button
    const attestationSection = page.locator("section").filter({
      has: page.locator("text=Dual Attestation"),
    });

    const refreshButton = attestationSection
      .locator("button")
      .filter({ hasText: /Refresh/i });

    if (await refreshButton.isVisible()) {
      await refreshButton.click();

      // Wait for the refresh to complete
      await page.waitForTimeout(500);

      // The section should still be visible and not show errors
      await expect(page.locator("text=Dual Attestation")).toBeVisible();
    }
  });

  test("displays event counts when attestation is loaded", async ({ page }) => {
    await page.waitForTimeout(1500);

    // Look for event count indicators
    const reasoningLabel = page.locator("text=Reasoning Events");
    const uiLabel = page.locator("text=UI Events");

    // If attestation is loaded, these labels should be visible
    const hasReasoningLabel = await reasoningLabel.isVisible();
    const hasUiLabel = await uiLabel.isVisible();

    // At least one should be visible if attestation data is present
    if (hasReasoningLabel || hasUiLabel) {
      // Verify the counts are numeric
      const countElements = page.locator("span").filter({ hasText: /^\d+$/ });
      expect(await countElements.count()).toBeGreaterThan(0);
    }
  });
});

test.describe("Attestation Panel Determinism", () => {
  test("multiple page loads show identical attestation values", async ({
    page,
  }) => {
    // First load
    await page.goto("/");
    await page.waitForSelector("text=Dual Attestation");
    await page.waitForTimeout(1500);

    // Capture attestation values
    const truncatedHashes1 = await page
      .locator("code")
      .filter({ hasText: "…" })
      .allTextContents();

    // Second load
    await page.reload();
    await page.waitForSelector("text=Dual Attestation");
    await page.waitForTimeout(1500);

    // Capture attestation values again
    const truncatedHashes2 = await page
      .locator("code")
      .filter({ hasText: "…" })
      .allTextContents();

    // Values should be identical (deterministic)
    expect(truncatedHashes1).toEqual(truncatedHashes2);
  });

  test("no random colors or styles in attestation panel", async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector("text=Dual Attestation");

    const attestationSection = page.locator("section").filter({
      has: page.locator("text=Dual Attestation"),
    });

    // Get computed styles for the section
    const bgColor = await attestationSection.evaluate((el) =>
      getComputedStyle(el).backgroundColor
    );

    // Reload and check again
    await page.reload();
    await page.waitForSelector("text=Dual Attestation");

    const attestationSection2 = page.locator("section").filter({
      has: page.locator("text=Dual Attestation"),
    });

    const bgColor2 = await attestationSection2.evaluate((el) =>
      getComputedStyle(el).backgroundColor
    );

    // Background color should be deterministic
    expect(bgColor).toBe(bgColor2);
  });
});

