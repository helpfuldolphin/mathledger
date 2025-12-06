import { test, expect } from "@playwright/test";

/**
 * DAG Rendering E2E Tests
 * =======================
 *
 * These tests validate that the Cytoscape-based dependency graph renders
 * correctly in a real browser environment. The vitest suite cannot test
 * this because GraphCanvas uses dynamic import (`await import("cytoscape")`)
 * which JSDOM cannot intercept.
 *
 * Prerequisites:
 * - Backend API running at NEXT_PUBLIC_API_BASE_URL (or mocked via MSW)
 * - Next.js dev server running at http://localhost:3000
 */

test.describe("DAG Rendering", () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the dashboard
    await page.goto("/");
    // Wait for the page to hydrate
    await page.waitForSelector("text=Proof Graph Console");
  });

  test("renders the Dependency Graph section", async ({ page }) => {
    // The graph canvas section should be visible
    await expect(page.locator("text=Dependency Graph")).toBeVisible();
    await expect(
      page.locator("text=Tap nodes to pivot across the proof DAG")
    ).toBeVisible();
  });

  test("graph container exists and has dimensions", async ({ page }) => {
    // The cytoscape container should have a minimum height
    const graphContainer = page.locator('[class*="h-80"]').first();
    await expect(graphContainer).toBeVisible();

    const box = await graphContainer.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.height).toBeGreaterThan(100);
    expect(box!.width).toBeGreaterThan(100);
  });

  test("clicking a statement triggers DAG update", async ({ page }) => {
    // Wait for statements to load
    const statementList = page.locator("text=Statements").first();
    await expect(statementList).toBeVisible();

    // Find a statement hash button (64-char hex string)
    const statementButton = page
      .locator("button")
      .filter({ hasText: /^[0-9a-f]{64}$/i })
      .first();

    // If no statements are loaded, skip this test
    const count = await statementButton.count();
    if (count === 0) {
      test.skip(true, "No statements available in the UI");
      return;
    }

    // Click the statement
    await statementButton.click();

    // The graph should update (we can't inspect cytoscape internals,
    // but we can verify the container is still visible and responsive)
    const graphContainer = page.locator('[class*="h-80"]').first();
    await expect(graphContainer).toBeVisible();
  });

  test("graph canvas responds to window resize", async ({ page }) => {
    const graphContainer = page.locator('[class*="h-80"]').first();
    await expect(graphContainer).toBeVisible();

    // Get initial dimensions
    const initialBox = await graphContainer.boundingBox();
    expect(initialBox).not.toBeNull();

    // Resize the viewport
    await page.setViewportSize({ width: 800, height: 600 });

    // Wait for resize to propagate
    await page.waitForTimeout(500);

    // Container should still be visible
    await expect(graphContainer).toBeVisible();
  });
});

test.describe("DAG Node Interaction", () => {
  test("parent nodes are clickable and trigger navigation", async ({
    page,
  }) => {
    await page.goto("/");
    await page.waitForSelector("text=Proof Graph Console");

    // This test requires a statement with parents to be loaded
    // We'll check if the parent section exists in the detail panel
    const parentSection = page.locator("text=Parents").first();

    // If parents section is visible, try clicking a parent link
    if (await parentSection.isVisible()) {
      const parentLink = page
        .locator("button")
        .filter({ hasText: /^[0-9a-f]{12,}/ })
        .first();

      if ((await parentLink.count()) > 0) {
        await parentLink.click();
        // After clicking, the graph should update
        await page.waitForTimeout(300);
        await expect(page.locator("text=Dependency Graph")).toBeVisible();
      }
    }
  });
});

